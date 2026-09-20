# RidgeVision AI — Research Evidence Package

This directory (`RidgeVision_Research_Evidence/`) contains the complete, scientifically audited factual record of the **RidgeVision AI** project, generated directly from the existing repository files, code, notebooks, and result logs.

## Purpose of this Evidence Package
This evidence package serves as the verified empirical foundation for composing a publication-ready academic research paper. Every parameter, metric, and finding recorded here is grounded strictly in existing code and data.

## Verification Rules Followed
1. **Zero Fabrication**: No experimental values, accuracies, or parameters have been invented, rounded, estimated, or improved.
2. **Fact Traceability**: Every number is traceable to an actual source file and line.
3. **Explicit Uncertainty**: Where information could not be verified from files, it is explicitly marked `NOT FOUND IN PROJECT FILES`.
4. **Distinction of Phases**: Clearly separates the prototype Phase 1 (8,000 images, image-level split) from the journal-grade Phase 2 (5,837 images, pHash cluster split).

## Files in this Directory
1. `00_AUDIT_REPORT.md`: Comprehensive executive audit report covering sections A through M.
2. `01_PROJECT_OVERVIEW.md`: System objectives, architecture overview, and technology stack.
3. `02_DATASET_AUDIT.md`: Complete dataset distribution, sub-cohort counts, and provenance analysis.
4. `03_PREPROCESSING_AUDIT.md`: Sequential 8-step enhancement pipeline and 30-dim texture extractor.
5. `04_DATA_SPLIT_AUDIT.md`: Detailed audit of image-level vs cluster-level data splitting and leakage controls.
6. `05_MODEL_ARCHITECTURE.md`: Complete layer-by-layer architectural tables and mathematical layer formulations.
7. `06_TRAINING_CONFIGURATION.md`: Hyperparameters, two-stage Adam optimizer recipes, and hardware specs.
8. `07_EXPERIMENTS_CATALOG.md`: Catalog of all 26 distinct experiments conducted in the project.
9. `08_RESULTS_MASTER.md`: Master table of verified experimental metrics, confusion matrices, and ANOVA effect sizes.
10. `09_BASELINE_COMPARISON.md`: 10-model baseline comparison table across 3-fold grouped cross-validation.
11. `10_FIGURES_AND_TABLES_CATALOG.md`: Catalog of all existing visual figures and documentation tables.
12. `11_REPRODUCIBILITY_AUDIT.md`: Independent reproducibility assessment (Complete / Partial / Missing).
13. `12_RESEARCH_RISKS_AND_MISSING_INFORMATION.md`: Critical analysis of biological disconnects and peer review vulnerabilities.
14. `13_RESEARCH_MASTER_SHEET.csv`: Structured machine-readable CSV of all project parameters and facts.
15. `14_RIDGEVISION_EXPERIMENT_SUMMARY.md`: Concise, structured summary of the current experimental framework.
16. `README.md`: This reference index.

> [!WARNING]
> **CRITICAL WARNING FOR RESEARCHERS**:  
> Do NOT modify or artificially inflate any values in these evidence files. Scientific accuracy and methodological transparency are paramount for peer review in IEEE, Elsevier, or Springer biomedical venues.
