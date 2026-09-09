from backend.ml.training.splits import (
    generate_shuffled_label_control,
    get_leaksafe_train_test_split,
)


def test_leaksafe_train_test_split():
    # Synthetic manifest with cluster IDs
    manifest = [
        {"path": f"img_{i}.png", "label": "A+", "cluster_id": f"cluster_{i // 2}"}
        for i in range(20)
    ]

    train, test = get_leaksafe_train_test_split(manifest, test_size=0.20, random_state=42)

    assert len(train) > 0
    assert len(test) > 0
    assert len(train) + len(test) == len(manifest)

    train_clusters = {r["cluster_id"] for r in train}
    test_clusters = {r["cluster_id"] for r in test}

    # Strict isolation between train and test clusters
    assert len(train_clusters.intersection(test_clusters)) == 0


def test_generate_shuffled_label_control():
    manifest = [
        {"path": f"img_{i}.png", "label": "A+" if i % 2 == 0 else "O-"}
        for i in range(50)
    ]

    shuffled = generate_shuffled_label_control(manifest, random_state=42)

    assert len(shuffled) == len(manifest)
    # Target values exist and labels are updated
    for r in shuffled:
        assert "label" in r
        assert "abo_label" in r
        assert "rh_label" in r
