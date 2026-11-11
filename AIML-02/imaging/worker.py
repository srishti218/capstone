"""Poll Orthanc's changes log and persist study results in SQLite."""
import hashlib
import json
import logging
import os
import time
from collections import defaultdict

import requests
from pydicom.errors import InvalidDicomError

from .dicom import UnsupportedStudy, compatible_body_part, is_frontal, read_dicom
from .inference import Inference, MissingWeights
from .store import Store

LOG = logging.getLogger(__name__)


class Orthanc:
    def __init__(self):
        self.base = os.getenv("ORTHANC_URL", "http://127.0.0.1:8042").rstrip("/")
        self.session = requests.Session()
        if os.getenv("ORTHANC_PASSWORD"):
            self.session.auth = os.getenv("ORTHANC_USERNAME", "radreview"), os.environ["ORTHANC_PASSWORD"]

    def get(self, path, **params):
        response = self.session.get(self.base + path, params=params, timeout=(5, 30))
        response.raise_for_status()
        return response

    def changes(self, since):
        return self.get("/changes", since=since, limit=100).json()

    def study_inputs(self, study):
        info = self.get(f"/studies/{study}").json()
        if not info["IsStable"]:
            return None
        groups = defaultdict(list)
        for instance in self.get(f"/studies/{study}/instances").json():
            tags = instance.get("MainDicomTags", {})
            if tags.get("SOPClassUID") and tags["SOPClassUID"] not in {
                "1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.1",
                "1.2.840.10008.5.1.4.1.1.1.1", "1.2.840.10008.5.1.4.1.1.1.1.1",
                "1.2.840.10008.5.1.4.1.1.7",
            }:
                continue
            instance_id = instance["ID"]
            try:
                ds = read_dicom(self.get(f"/instances/{instance_id}/file").content)
            except InvalidDicomError:
                LOG.warning("Skipping invalid DICOM instance %s", instance_id)
                continue
            if is_frontal(ds):
                groups["classification", instance_id].append((instance_id, ds))
            if (
                ds.get("Modality") == "CT" and ds.get("SeriesInstanceUID")
                and compatible_body_part(ds) and "LOCALIZER" not in (ds.get("ImageType") or ())
            ):
                groups["segmentation", str(ds.SeriesInstanceUID)].append((instance_id, ds))
        # A transfer may have started while the files were being downloaded.
        if not self.get(f"/studies/{study}").json()["IsStable"]:
            return None
        return groups


def process_study(orthanc, store, inference, study):
    groups = orthanc.study_inputs(study)
    if groups is None:
        return "retry"
    saved = missing = False
    for (kind, _), items in groups.items():
        ids, datasets = zip(*items)
        try:
            model, version, digest = inference.model(kind)
            result_id = hashlib.sha256(
                json.dumps([study, kind, sorted(ids), version, digest]).encode()
            ).hexdigest()[:32]
            if not store.has_result(result_id):
                if kind == "classification":
                    metadata, frames = inference.cxr(model, datasets[0])
                else:
                    metadata, frames = inference.ct(model, datasets)
                metadata.update(study=study, model=version, checkpoint=digest, demo=False)
                store.save_result(result_id, metadata, frames)
            saved = True
        except MissingWeights:
            missing = True
        except UnsupportedStudy as exc:
            LOG.info("Skipping unsupported %s input for %s (%s)", kind, study, exc)
    if missing:
        return "awaiting weights"
    return "done" if saved else "unsupported"


def poll(orthanc, store, inference):
    for _ in range(10):
        page = orthanc.changes(store.cursor())
        changes = [
            (item["ID"], item["Seq"]) for item in page["Changes"]
            if item["ChangeType"] == "StableStudy" and item["ResourceType"] == "Study"
        ]
        store.enqueue(changes, page["Last"])
        if page["Done"]:
            break
    for job in store.due_jobs(int(time.time())):
        study = job["study"]
        try:
            status = process_study(orthanc, store, inference, study)
        except requests.HTTPError as exc:
            deleted = (
                exc.response.status_code == 404
                and exc.request.url == f"{orthanc.base}/studies/{study}"
            )
            status = "deleted" if deleted else "retry"
            LOG.warning("Orthanc returned %s while processing %s", exc.response.status_code, study)
        except Exception:
            status = "retry"
            LOG.exception("Could not process study %s", study)
        delay = {"retry": min(300, 5 * 2 ** min(job["attempts"], 6)), "awaiting weights": 30}
        store.update_job(job, status, int(time.time()) + delay[status] if status in delay else 0)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    orthanc, store, inference = Orthanc(), Store(), Inference()
    LOG.info("Polling Orthanc at %s", orthanc.base)
    while True:
        try:
            poll(orthanc, store, inference)
        except requests.RequestException:
            LOG.exception("Orthanc polling failed. Retrying.")
        time.sleep(3)


if __name__ == "__main__":
    main()
