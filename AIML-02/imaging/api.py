"""Same-origin viewer and review API."""
import os
import sqlite3
from pathlib import Path
from uuid import UUID

from flask import Flask, Response, jsonify, request, send_from_directory

from .store import Store


def create_app(db_path=None, seed=None):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 16_384
    store = Store(db_path)
    if seed is None:
        seed = os.getenv("RADREVIEW_DEMO", "1") == "1"
    if seed:
        from .demo import seed_demo
        seed_demo(store)
    dist = Path(os.getenv("RADREVIEW_VIEWER", Path(__file__).resolve().parents[1] / "viewer" / "dist"))

    @app.after_request
    def headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.get("/")
    def index():
        return send_from_directory(dist, "index.html")

    @app.get("/assets/<path:name>")
    def asset(name):
        return send_from_directory(dist / "assets", name)

    @app.get("/api/health")
    def health():
        return jsonify(status="ok", jobs=store.job_counts())

    @app.get("/api/results")
    def results():
        return jsonify(store.results())

    @app.get("/api/results/<result_id>/frames/<int:position>/<layer>")
    def frame(result_id, position, layer):
        pixels = store.frame(result_id, position, layer) if layer in {"image", "overlay"} else None
        if pixels is None:
            return jsonify(error="Image frame not found."), 404
        return Response(pixels, mimetype="image/png", headers={"Cache-Control": "private, max-age=86400"})

    @app.get("/api/audit")
    def audit():
        return jsonify(store.audit(request.args.get("result_id")))

    @app.post("/api/results/<result_id>/reviews")
    def review(result_id):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify(error="A JSON decision is required."), 400
        actor = payload.get("actor")
        note = payload.get("note", "")
        decision = payload.get("decision")
        try:
            request_id = str(UUID(str(payload.get("request_id"))))
        except ValueError:
            return jsonify(error="A valid decision request ID is required."), 400
        if (
            not isinstance(actor, str) or not 1 <= len(actor.strip()) <= 80
            or not isinstance(note, str) or len(note) > 500
            or decision not in ("approved", "rejected")
        ):
            return jsonify(error="Enter a reviewer name and choose Approve or Reject. Notes allow 500 characters."), 400
        try:
            entry = store.review(result_id, request_id, actor.strip(), decision, note.strip())
        except ValueError as error:
            return jsonify(error=str(error)), 409
        except sqlite3.IntegrityError:
            return jsonify(error="Result not found."), 404
        return jsonify(entry), 201

    return app
