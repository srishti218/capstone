# RadReview

**For education only, not clinical use.** RadReview is a local radiology AI review workspace for public, de-identified datasets.

| Workflow | Public data | Output |
| --- | --- | --- |
| Frontal chest radiograph | [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data) | ImageNet-initialized DenseNet-121, an opacity score and Grad-CAM for the predicted class |
| Axial chest CT | [Medical Segmentation Decathlon Task06 Lung](https://registry.opendata.aws/msd/) | MONAI 3D U-Net and a voxel mask for a candidate lung tumor |

The RSNA labels describe opacity on 2D radiographs and include bounding boxes. The 3D model uses annotated CT volumes from a separate dataset and targets lung tumors. [DICOM notes](docs/dicom.md) · [Model card and bias](docs/model-card.md)

## Open the viewer

With Docker running, build the viewer and Orthanc stack. The first build downloads the container images and inference dependencies.

```bash
mkdir -p weights
docker compose up --build -d
docker compose ps
```

Open [RadReview](http://localhost:8080/) to review the included sample case. Its score and heatmap are illustrative. Enter a reviewer name, submit an approval or rejection, and open **Audit log** to see the reviewer, UTC timestamp, and note. The latest decision sets the worklist status. Every submitted decision remains in the log.

For the sample viewer alone, run `docker compose up --build viewer`. You can also start the viewer without Docker using Python 3.11 and Node 22.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
npm --prefix viewer ci
npm --prefix viewer run build
gunicorn --bind 127.0.0.1:8080 'imaging.api:create_app()'
```

If Compose reports `unexpected end of JSON input` while looking up images, run `docker version`. If it prints `EOF` in place of the Server section, check free space on the Mac with `df -h .`. A `no space left on device` message in Docker Desktop's logs means you need to free host storage before restarting. On macOS, run `docker desktop restart`. If the engine is already down and the restart times out while stopping Desktop, run `docker desktop stop --force`, then `docker desktop start`. Wait for `docker version` to show the Server section before retrying Compose. [Docker Desktop troubleshooting](https://docs.docker.com/desktop/troubleshoot-and-support/troubleshoot/)

## Download and version the training data

1. Sign in to Kaggle, accept the [competition rules](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules), and extract `stage_2_train_labels.csv` and `stage_2_train_images/` under `data/rsna/`. Use your own Kaggle credentials if downloading with its CLI.
2. Download [Task06 Lung](https://medicaldecathlon.com/dataaws/) from the official mirror and extract it so that `data/Task06_Lung/imagesTr/` and `data/Task06_Lung/labelsTr/` exist.
3. Allow at least 45 GB for archives, extracted files, generated caches, and Docker images. A CUDA GPU is useful for full training. Use Python 3.11 and install the training dependencies and DVC.

```bash
python -m pip install -r requirements-ml.txt -r requirements-dvc.txt
dvc add data/rsna data/Task06_Lung
dvc repro train_cxr
dvc repro train_ct
dvc metrics show
```

DVC records the raw dataset fingerprints, preprocessing and training dependencies, split caches, and checkpoint hashes. After a successful run, record the two `.dvc` pointer files and `dvc.lock` in Git. Store raw data and artifacts in a private DVC remote only where the dataset terms allow it. DVC metrics are written under `reports/`. The default CXR run downloads ImageNet weights from torchvision. The CT model is trained from scratch.

## Send a study to Orthanc

The worker reads `weights/classifier.pt` and `weights/segmenter.pt`. Copy checkpoints into the mounted directory before starting the worker, or restart the worker after replacing them. Results that are waiting for their first checkpoint retry automatically. Existing results preserve their checkpoint identity and audit history.

Upload a public frontal DICOM. Orthanc also accepts a series of supported chest CT slices uploaded individually. A NIfTI file is a training input and cannot be uploaded as DICOM.

```bash
curl --fail-with-body -H 'Content-Type: application/dicom' \
  --data-binary @data/rsna/stage_2_train_images/YOUR_CASE.dcm \
  http://127.0.0.1:8042/instances
docker compose logs -f worker
```

Orthanc exposes its explorer on [localhost port 8042](http://localhost:8042/), and DICOM C-STORE on port 4242 using AE title `RADREVIEW`. The worker queues stable study events, retries transient failures, and writes PNG layers, scores, and model provenance to SQLite. The named Docker volumes hold the Orthanc files and the review database. The [health endpoint](http://localhost:8080/api/health) shows job counts, including studies waiting for checkpoints.

The stack binds ports to loopback and is intended for one local operator. Reviewer names are self-reported. Use public or synthetic data and keep the stack local. Use `RADREVIEW_DEMO=0 docker compose up -d` for a fresh deployment without a seeded sample. A sample already stored in the persistent database remains available.

## Verify the review workflow

Run `python -m unittest discover -s tests -v` to exercise the image endpoints, review retries, persistent audit history, and study queue. Run `npm --prefix viewer run build` to compile the viewer.
