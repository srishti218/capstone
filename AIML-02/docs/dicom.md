# DICOM in this project

A DICOM object combines metadata and an encoded pixel payload. A study groups one or more series, and a series groups related instances. Each has an identifier. `StudyInstanceUID`, `SeriesInstanceUID`, and `SOPInstanceUID` identify these levels. Orthanc assigns separate REST resource IDs. The file's transfer syntax controls how the pixel bytes are decoded.

## Chest radiographs

The worker accepts single-frame, grayscale CR and DX images tagged with an AP or PA projection. It decodes pixels with pydicom, applies the modality and VOI transforms, and scales the non-padding intensities using their first and 99th percentiles. It inverts MONOCHROME1 to match the display polarity. The classifier uses a 224 by 224 preview and ImageNet channel normalization. Grad-CAM is resized back to the preview's original grid.

## CT geometry

CT is reconstructed from single-frame, axis-aligned axial slices with consistent pixel spacing, frame of reference, and image dimensions. The worker sorts slices by `ImagePositionPatient` and verifies uniform spacing. It converts stored values to Hounsfield units using rescale slope and intercept. It clips intensities to −1000 through 400 HU, rescales to zero through one, and resamples to approximately 2 mm isotropic voxels. The mask maps back to the original slice grid. For training, the NIfTI images and labels must share an axis-aligned affine and are converted from canonical RAS to the same LPS ordering.

Upload only chest studies. The worker excludes other body regions when the DICOM body-part tag is populated. Support is limited to transfer syntaxes that pydicom can decode with the installed runtime. Enhanced or multiframe CT, localizers, gantry tilt, irregular spacing, and color images are outside the supported scope. The viewer displays fixed-window PNG previews and stores the model output separately in SQLite. The original DICOM is held in Orthanc.

[Orthanc DICOM format primer](https://orthanc.uclouvain.be/book/dicom-guide.html) · [Orthanc REST API](https://orthanc.uclouvain.be/book/users/rest.html)
