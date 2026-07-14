# Local data layout

Datasets and generated feature matrices are deliberately excluded from Git because of size, licensing, and privacy considerations.

Expected local paths include:

```text
data/
├── external/skyfinder/                  SkyFinder source dataset
├── prl_images/                          127 original PRL images
├── prl/
│   ├── metadata/prl_dataset_clean.xlsx  Clean PRL metadata and labels
│   ├── features/                        Generated DINOv2 embeddings
│   └── processed/                       Generated PRL fusion matrices
├── processed/                           Generated SkyFinder training arrays
└── merged/                              Generated merged source tables
```

Generate PRL embeddings and downstream artifacts with:

```powershell
python run_prl_pipeline.py --extract-features --force-features --require-images
```

Do not commit raw images, embeddings, `.npy` arrays, or private metadata unless dataset publication rights and location/privacy constraints have been reviewed.
