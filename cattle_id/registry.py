import json
from pathlib import Path
from typing import Protocol


class Registry(Protocol):
    def put(self, cattle_id: str, details: dict) -> None: ...
    def get(self, cattle_id: str) -> dict | None: ...
    def find_by_tag(self, tag_number: str) -> str | None: ...  # cattle_id whose details['tag_number'] matches


class JsonRegistry:
    """Local dev registry. Swap for FirestoreRegistry (same two methods) in the integrated app."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def put(self, cattle_id: str, details: dict) -> None:
        self.data[cattle_id] = details
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, cattle_id: str) -> dict | None:
        return self.data.get(cattle_id)

    def find_by_tag(self, tag_number: str) -> str | None:
        return next((cid for cid, d in self.data.items() if d.get("tag_number") == tag_number), None)


class FirestoreRegistry:
    def __init__(self, collection: str = "cattle", project: str | None = None):
        from google.cloud import firestore

        self.col = firestore.Client(project=project).collection(collection)

    def put(self, cattle_id: str, details: dict) -> None:
        self.col.document(cattle_id).set(details)

    def get(self, cattle_id: str) -> dict | None:
        doc = self.col.document(cattle_id).get()
        return doc.to_dict() if doc.exists else None

    def find_by_tag(self, tag_number: str) -> str | None:
        docs = self.col.where("tag_number", "==", tag_number).limit(1).stream()
        return next((d.id for d in docs), None)
