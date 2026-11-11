# RadReview model card

**For education only, not clinical use.** Both model workflows require trained checkpoints. No public-data training results or validated performance are included in this workspace.

## DenseNet-121 chest X-ray classifier

- **Target.** The RSNA binary label indicates radiographic findings suggestive of pneumonia. It reflects image interpretation and does not establish infection. The score is an uncalibrated sigmoid output at a demonstration threshold of 0.5.
- **Training.** The pipeline initializes DenseNet-121 from ImageNet and fine-tunes with class-weighted binary cross-entropy. It applies affine rotation, translation, and scaling to training images. The split is deterministic and stratified by the challenge's image-level `patientId`, grouping multiple annotation rows for an image. The challenge identifier cannot establish linkage across acquisitions beyond the information released in the dataset.
- **Evaluation.** The validation ROC AUC selects the checkpoint. The held-out report gives ROC AUC, average precision, sensitivity, specificity, and Brier score overall and by AP/PA projection, recorded sex, and age band where available. Subgroup counts matter, and tiny groups may yield unstable estimates. The Brier score measures probabilistic error and is not a calibration procedure.
- **Interpretation.** Grad-CAM highlights positive contribution to the predicted class at the final DenseNet feature map. It is spatially coarse and can emphasize text, devices, positioning, or other confounders.
- **Bias and generalization.** RSNA uses frontal chest X-rays derived from NIH ChestX-ray data. The label mix, acquisition setting, image transformations, and projection may differ from other sites and populations. AP/PA can correlate with patient positioning and care setting. Negative cases can have other abnormalities. Performance on lateral views, pediatric populations, or other clinical settings has not been established. [RSNA dataset description](https://atlas.rsna.org/cards/74777c92-9b0b-4f06-a788-71a8a12c4e93)

## MONAI 3D U-Net chest CT segmenter

- **Target.** Task06 Lung provides lung tumor masks for volumetric CT. The model predicts a binary candidate mask using a 0.5 voxel threshold. The displayed number averages the scores inside selected voxels and is undefined if the mask is empty.
- **Training.** The pipeline splits by volume, samples 3D patches with foreground oversampling, and augments with joint image/mask flips and intensity offsets. It optimizes Dice plus cross-entropy. Sliding-window inference and validation use the full resampled volume.
- **Evaluation.** Mean validation Dice selects the checkpoint. The held-out report includes per-case and mean Dice. The cohort is small, drawn from patients with lung cancer, and cannot establish false-positive rates on tumor-free CT. Scanner, slice thickness, reconstruction, and population shifts may change performance. [Medical Segmentation Decathlon paper](https://doi.org/10.1038/s41467-022-30695-9)

## Review and storage

Approve and Reject record a reviewer's judgment of a model result. SQLite records each event against the original result and checkpoint digest and prevents app-level edits or deletes. The local operator controls the database, and reviewer names are self-reported. The sample case uses a resized copy of the CC0 chest radiograph by Stillwaterising, with a hand-drawn heatmap and a fixed example score. [Wikimedia Commons image record](https://commons.wikimedia.org/wiki/File:Chest_Xray_PA_3-8-2010.png)
