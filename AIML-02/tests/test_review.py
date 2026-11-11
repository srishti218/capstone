"""Exercise the educational review workflow and its persistence guarantees."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

from imaging.api import create_app
from imaging.store import Store


class ReviewTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "review.sqlite3"
        self.client = create_app(self.path, seed=True).test_client()
        result, = self.client.get("/api/results").json
        self.result_id = result["id"]
        self.base = f"/api/results/{self.result_id}"
        self.payload = {"request_id": str(uuid4()), "actor": "Reviewer", "decision": "approved"}

    def test_decisions_survive_retries_and_restart(self):
        first = self.client.post(f"{self.base}/reviews", json=self.payload)
        self.assertEqual(first.status_code, 201)
        self.assertEqual(self.client.post(f"{self.base}/reviews", json=self.payload).json, first.json)

        self.payload["decision"] = "rejected"
        self.assertEqual(self.client.post(f"{self.base}/reviews", json=self.payload).status_code, 409)
        self.payload["request_id"] = str(uuid4())
        self.assertEqual(self.client.post(f"{self.base}/reviews", json=self.payload).status_code, 201)

        fresh = create_app(self.path, seed=True).test_client()
        self.assertEqual(fresh.get("/api/results").json[0]["status"], "rejected")
        self.assertEqual([item["decision"] for item in fresh.get("/api/audit").json], ["rejected", "approved"])
        with self.assertRaisesRegex(sqlite3.IntegrityError, "append-only"):
            with Store(self.path).connect() as db:
                db.execute("DELETE FROM decisions")

    def test_scan_layers_and_missing_frame(self):
        for layer in ("image", "overlay"):
            response = self.client.get(f"{self.base}/frames/0/{layer}")
            self.assertEqual(response.mimetype, "image/png")
            self.assertTrue(response.data.startswith(b"\x89PNG"))
        self.assertEqual(self.client.get(f"{self.base}/frames/1/image").status_code, 404)

    def test_new_study_event_supersedes_stale_job_updates(self):
        store = Store(self.path)
        store.enqueue([("study", 5)], 8)
        first, = store.due_jobs(0)
        store.enqueue([("study", 9)], 10)
        store.update_job(first, "done")
        store.enqueue([("study", 5)], 8)
        second, = store.due_jobs(0)
        self.assertEqual(second["seq"], 9)
        self.assertEqual(Store(self.path).cursor(), 10)


if __name__ == "__main__":
    unittest.main()
