import tensorflow as tf

from backend.core.config import CLASS_LABELS

ABO_CLASSES = ["A", "B", "AB", "O"]
RH_CLASSES = ["+", "-"]


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class RidgeOrientationField(tf.keras.layers.Layer):
    """Deterministic, non-trainable ridge orientation field."""

    def __init__(self, block_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)[:, :, None, None]
        self.sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=tf.float32)[:, :, None, None]

    def call(self, inputs):
        gx = tf.nn.conv2d(inputs, self.sobel_x, strides=1, padding="SAME")
        gy = tf.nn.conv2d(inputs, self.sobel_y, strides=1, padding="SAME")
        gxx, gyy, gxy = gx * gx, gy * gy, gx * gy
        pool = lambda t: tf.nn.avg_pool2d(t, ksize=self.block_size, strides=self.block_size, padding="VALID")
        vxx, vyy, vxy = pool(gxx), pool(gyy), pool(gxy)
        numerator = 2.0 * vxy
        denominator = vxx - vyy
        theta2 = tf.atan2(numerator, denominator)
        cos2, sin2 = tf.cos(theta2), tf.sin(theta2)
        energy = tf.sqrt(numerator**2 + denominator**2)
        coherence = tf.clip_by_value(energy / (vxx + vyy + 1e-6), 0.0, 1.0)
        return tf.concat([cos2, sin2, coherence], axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"block_size": self.block_size})
        return config


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class ROAM(tf.keras.layers.Layer):
    """Ridge Orientation Attention Module."""

    def __init__(self, reduction: int = 4, use_channel_gate: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.reduction = reduction
        self.use_channel_gate = use_channel_gate

    def build(self, input_shapes):
        feature_shape, _ = input_shapes
        channels = int(feature_shape[-1])
        reduced = max(channels // self.reduction, 8)
        self.orientation_proj = tf.keras.layers.Conv2D(reduced, 3, padding="same", activation="relu")
        self.spatial_gate = tf.keras.layers.Conv2D(1, 1, padding="same", activation="sigmoid")
        if self.use_channel_gate:
            self.channel_squeeze = tf.keras.layers.Dense(reduced, activation="relu")
            self.channel_excite = tf.keras.layers.Dense(channels, activation="sigmoid")
        self.resize_target = (int(feature_shape[1]), int(feature_shape[2]))

    def call(self, inputs):
        features, orientation_field = inputs
        orientation_resized = tf.image.resize(orientation_field, self.resize_target, method="bilinear")
        spatial_attention = self.spatial_gate(self.orientation_proj(orientation_resized))
        if self.use_channel_gate:
            channel_stats = tf.reduce_mean(features, axis=[1, 2])
            channel_attention = self.channel_excite(self.channel_squeeze(channel_stats))[:, None, None, :]
        else:
            channel_attention = 1.0
        attended = features * spatial_attention * channel_attention
        return attended, spatial_attention

    def get_config(self):
        config = super().get_config()
        config.update({"reduction": self.reduction, "use_channel_gate": self.use_channel_gate})
        return config


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class AdaptiveGatedFusion(tf.keras.layers.Layer):
    """Adaptive Gated Fusion between Ridge and Appearance branches."""

    def build(self, input_shapes):
        a_shape, b_shape = input_shapes
        dim = max(int(a_shape[-1]), int(b_shape[-1]))
        self.proj_a = tf.keras.layers.Dense(dim)
        self.proj_b = tf.keras.layers.Dense(dim)
        self.gate_dense = tf.keras.layers.Dense(dim, activation="sigmoid")

    def call(self, inputs):
        branch_a, branch_b = inputs
        a, b = self.proj_a(branch_a), self.proj_b(branch_b)
        gate = self.gate_dense(tf.concat([a, b], axis=-1))
        return gate * a + (1.0 - gate) * b, gate


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class ChannelAttention(tf.keras.layers.Layer):
    def __init__(self, ratio: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced = max(channels // self.ratio, 1)
        self.shared_dense_1 = tf.keras.layers.Dense(reduced, activation="relu")
        self.shared_dense_2 = tf.keras.layers.Dense(channels)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        return inputs * tf.nn.sigmoid(avg_out + max_out)

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv2D(
            1,
            kernel_size=kernel_size,
            padding="same",
            activation="sigmoid",
        )

    def build(self, input_shape):
        batch = input_shape[0] if len(input_shape) >= 1 else None
        h = input_shape[1] if len(input_shape) >= 2 else None
        w = input_shape[2] if len(input_shape) >= 3 else None
        self.conv.build((batch, h, w, 2))
        super().build(input_shape)

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        return inputs * self.conv(concat)

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


@tf.keras.utils.register_keras_serializable(package="RidgeVision")
class CBAM(tf.keras.layers.Layer):
    def __init__(self, ratio: int = 8, kernel_size: int = 7, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.channel_attention = ChannelAttention(ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def build(self, input_shape):
        self.channel_attention.build(input_shape)
        self.spatial_attention.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        x = self.channel_attention(inputs)
        return self.spatial_attention(x)

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config



def build_cbam_block(feature_map, ratio: int = 8, kernel_size: int = 7):
    """Convolutional Block Attention Module (CBAM) implementation."""
    cbam_layer = CBAM(ratio=ratio, kernel_size=kernel_size, name="cbam_block")
    return cbam_layer(feature_map)


def build_leaksafe_cgn_model(
    image_shape: tuple[int, int, int] = (224, 224, 3),
    texture_dim: int = 30,
    include_hierarchical_heads: bool = True,
):
    """RidgeVision v2: LeakSafe-CGN Architecture.

    Incorporates:
      1. EfficientNetB0 deep feature backbone with end-to-end trainable CBAM attention.
      2. Unified multi-scale texture feature projection.
      3. Decoupled hierarchical biological prediction heads:
         - 4-class ABO Group head (A, B, AB, O)
         - Binary Rh Factor head (+, -)
         - Backward-compatible 8-class flat blood group head.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to build the LeakSafe-CGN model.") from exc

    image_input = tf.keras.Input(shape=image_shape, name="image_input")
    texture_input = tf.keras.Input(shape=(texture_dim,), name="texture_input")

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=image_input,
    )
    backbone.trainable = True

    # Feature map from backbone top conv
    conv_features = backbone.output

    # Trainable CBAM Attention
    attended_features = build_cbam_block(conv_features, ratio=8, kernel_size=7)

    # Pooling
    cnn_pooled = tf.keras.layers.GlobalAveragePooling2D(name="cnn_gap")(attended_features)

    # Branch B: Texture projection
    texture_proj = tf.keras.layers.Dense(64, activation="relu", name="texture_projection")(texture_input)
    texture_proj = tf.keras.layers.LayerNormalization(name="texture_norm")(texture_proj)

    # Gated Multimodal Fusion
    fused = tf.keras.layers.Concatenate(name="multimodal_fusion")([cnn_pooled, texture_proj])
    gate = tf.keras.layers.Dense(fused.shape[-1], activation="sigmoid", name="fusion_gate")(fused)
    fused_attended = tf.keras.layers.Multiply(name="gated_fusion")([fused, gate])

    shared_rep = tf.keras.layers.Dense(256, activation="relu", name="shared_representation")(fused_attended)
    shared_rep = tf.keras.layers.Dropout(0.35, name="shared_dropout")(shared_rep)

    if include_hierarchical_heads:
        # Hierarchical biological heads
        abo_output = tf.keras.layers.Dense(4, activation="softmax", name="abo_group")(shared_rep)
        rh_output = tf.keras.layers.Dense(1, activation="sigmoid", name="rh_factor")(shared_rep)
        flat_output = tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax", name="blood_group")(shared_rep)

        model = tf.keras.Model(
            inputs=[image_input, texture_input],
            outputs={
                "abo_group": abo_output,
                "rh_factor": rh_output,
                "blood_group": flat_output,
            },
            name="ridgevision_v2_leaksafe_cgn",
        )
    else:
        flat_output = tf.keras.layers.Dense(len(CLASS_LABELS), activation="softmax", name="blood_group")(shared_rep)
        model = tf.keras.Model(
            inputs=[image_input, texture_input],
            outputs=flat_output,
            name="ridgevision_v2_flat",
        )

    return model


def build_fusion_model(
    image_shape: tuple[int, int, int] = (224, 224, 3),
    texture_dim: int = 30,
    hierarchical: bool = False,
):
    """Backward-compatible model builder."""
    if hierarchical:
        return build_leaksafe_cgn_model(
            image_shape=image_shape,
            texture_dim=texture_dim,
            include_hierarchical_heads=True,
        )
    return build_leaksafe_cgn_model(
        image_shape=image_shape,
        texture_dim=texture_dim,
        include_hierarchical_heads=False,
    )


def get_multitask_loss_weights(
    lambda_abo: float = 0.5,
    lambda_rh: float = 0.3,
    lambda_flat: float = 0.2,
) -> dict[str, float]:
    """Joint multi-task loss weighting dictionary."""
    return {
        "abo_group": lambda_abo,
        "rh_factor": lambda_rh,
        "blood_group": lambda_flat,
    }


def build_ridgevision_net(
    img_size: int = 224,
    num_classes: int = 8,
    dropout_rate: float = 0.35,
    trainable_backbone_layers: int = 40,
    use_orientation_field: bool = True,
    use_channel_gate: bool = True,
    fusion_mode: str = "adaptive",
    use_ridge_branch: bool = True,
    use_appearance_branch: bool = True,
    name: str = "ridgevision_net",
):
    """RidgeVisionNet v2 architecture with RidgeOrientationField, ROAM, and AdaptiveGatedFusion."""
    assert use_ridge_branch or use_appearance_branch
    image_input = tf.keras.Input(shape=(img_size, img_size, 3), name="fingerprint_image")
    efficientnet_input = tf.keras.layers.Rescaling(255.0, name="undo_pipeline_rescale_for_efficientnet")(image_input)
    backbone = tf.keras.applications.EfficientNetB0(weights=None, include_top=False, input_tensor=efficientnet_input)
    for layer in backbone.layers:
        layer.trainable = False
    if trainable_backbone_layers > 0:
        for layer in backbone.layers[-trainable_backbone_layers:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True

    branches = []
    if use_ridge_branch:
        mid_features = backbone.get_layer("block6a_expand_activation").output
        if use_orientation_field:
            grayscale = tf.keras.layers.Lambda(
                lambda x: tf.image.rgb_to_grayscale(x),
                output_shape=(img_size, img_size, 1),
                name="to_grayscale",
            )(image_input)
            orientation_field = RidgeOrientationField(block_size=8, dtype="float32")(grayscale)
        else:
            orientation_field = tf.keras.layers.Lambda(
                lambda x: tf.ones((tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 3)),
                output_shape=(None, None, 3),
                name="dummy_orientation_field",
            )(mid_features)
        roam = ROAM(use_channel_gate=use_channel_gate)
        attended_mid, spatial_attention = roam([mid_features, orientation_field])
        ridge_branch = tf.keras.layers.GlobalAveragePooling2D()(attended_mid)
        branches.append(ridge_branch)
    else:
        spatial_attention = None

    if use_appearance_branch:
        appearance_branch = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        branches.append(appearance_branch)

    if len(branches) == 1:
        fused = branches[0]
    elif fusion_mode == "adaptive":
        fused, _ = AdaptiveGatedFusion()(branches)
    elif fusion_mode == "static_average":
        dim = 256
        projected = [tf.keras.layers.Dense(dim)(b) for b in branches]
        fused = tf.keras.layers.Average()(projected)
    elif fusion_mode == "concat":
        fused = tf.keras.layers.Concatenate()(branches)
    else:
        raise ValueError(fusion_mode)

    hidden = tf.keras.layers.Dense(256, activation="relu")(fused)
    hidden = tf.keras.layers.Dropout(dropout_rate, name="mc_dropout")(hidden)
    logits = tf.keras.layers.Dense(num_classes, name="logits", dtype="float32")(hidden)
    probs = tf.keras.layers.Softmax(name="blood_group", dtype="float32")(logits)

    outputs = {"blood_group": probs, "logits": logits}
    if spatial_attention is not None:
        outputs["attention_map"] = spatial_attention
    return tf.keras.Model(inputs=image_input, outputs=outputs, name=name)
