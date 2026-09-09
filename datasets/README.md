# Dataset Notes

Use these sources as separate experiment inputs:

- Labeled ABO/Rh dataset: <https://www.kaggle.com/datasets/rajumavinmar/finger-print-based-blood-group-dataset>
- Original unlabeled SOCOFing dataset: <https://www.kaggle.com/datasets/ruizgara/socofing>

Recommended local layout:

```text
datasets/
|-- raw/
|   |-- blood_group_labeled/
|   `-- socofing/
|-- interim/
`-- processed/
    |-- train/
    |-- val/
    `-- test/
```

Keep downloaded Kaggle files out of git. Use the labeled dataset for supervised classification and SOCOFing for augmentation, ridge-quality benchmarking, self-supervised pretraining, or domain adaptation experiments.
