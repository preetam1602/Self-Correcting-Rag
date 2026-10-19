"""The HTTP API against the fake LLM and retriever: auth, scoping, errors, SSE and jobs."""
import time
from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
jwt = pytest.importorskip("jwt")

from fastapi.testclient import TestClient

from selfrag.api.app import create_app
from selfrag.api.settings import ApiSettings
from selfrag.critics.critic import Critic
from selfrag.ingestion.pipeline import IngestResult
from selfrag.orchestrator.graph import SelfRAG, SelfRAGConfig
from selfrag.tests.unit.test_orchestrator import GOOD, FakeLLM, FakeRetriever

ADMIN_KEY, SECRET = "admin-key", "jwt-secret-for-tests-only-32-bytes!"


class FakePipeline:
    def __init__(self):
        self.calls = []

    async def ingest_file(self, path, *, tenant_id, collection, acl):
        self.calls.append((path, tenant_id, collection, acl))
        if path.name.startswith("bad"):
            raise ValueError("cannot parse")
        return IngestResult("doc-" + path.stem, str(path), "ready", 3, 0, 0)


class FakeServices:
    def __init__(self):
        llm = FakeLLM()
        self.retriever = FakeRetriever([GOOD])
        self.rags = {m: SelfRAG(Critic(llm), llm, llm, self.retriever,
                                SelfRAGConfig(segment_unit=m)) for m in ("sentence", "answer")}
        self.default_mode = "answer"
        self.pipeline = FakePipeline()
        self.deleted = []

    async def ready(self):
        return {"qdrant": True, "postgres": True}

    async def delete_document(self, doc_id):
        self.deleted.append(doc_id)
        return doc_id == "known"

    async def close(self):
        pass


@pytest.fixture
def api(tmp_path):
    services = FakeServices()

    async def factory():
        return services

    settings = ApiSettings(_env_file=None, admin_api_key=ADMIN_KEY, jwt_secret=SECRET,
                           upload_dir=tmp_path)
    with TestClient(create_app(factory, settings)) as client:
        yield client, services


def user(tenant="acme", groups=("hr",), **extra):
    claims = {"tenant_id": tenant, "groups": list(groups),
              "exp": datetime.now(timezone.utc) + timedelta(minutes=5), **extra}
    return {"Authorization": f"Bearer {jwt.encode(claims, SECRET, algorithm='HS256')}"}


ADMIN = {"X-API-Key": ADMIN_KEY}
QUESTION = {"question": "What is the notice period?"}


def test_health_and_readiness(api):
    client, _ = api
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["status"] == "ready"


def test_answer_requires_valid_credentials(api):
    client, _ = api
    assert client.post("/v1/answer", json=QUESTION).status_code == 401
    assert client.post("/v1/answer", json=QUESTION, headers={"X-API-Key": "nope"}).status_code == 401
    expired = user(exp=datetime.now(timezone.utc) - timedelta(minutes=1))
    assert client.post("/v1/answer", json=QUESTION, headers=expired).status_code == 401


def test_user_answer_is_scoped_by_token(api):
    client, services = api
    r = client.post("/v1/answer", json={**QUESTION, "filters": {"collection": "hr-policies"}},
                    headers=user())
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["segments"][0]["citation"]["title"] == "Good doc"
    filters = services.retriever.calls[0][2]
    assert (filters.tenant_id, filters.collection, filters.acl_groups) == \
        ("acme", "hr-policies", ("hr",))


def test_user_cannot_widen_scope_or_override(api):
    client, _ = api
    for extra in ({"filters": {"tenant_id": "other"}}, {"filters": {"acl_groups": ["execs"]}},
                  {"overrides": {"k": 3}}):
        assert client.post("/v1/answer", json={**QUESTION, **extra},
                           headers=user()).status_code == 403


def test_admin_overrides_and_validation(api):
    client, services = api
    r = client.post("/v1/answer", headers=ADMIN, json={
        **QUESTION, "mode": "sentence", "overrides": {"k": 3},
        "filters": {"tenant_id": "acme", "acl_groups": ["hr"]}})
    assert r.status_code == 200
    assert services.retriever.calls[0][2].tenant_id == "acme"
    assert client.post("/v1/answer", headers=ADMIN,
                       json={**QUESTION, "overrides": {"k": 99}}).status_code == 400
    assert client.post("/v1/answer", headers=ADMIN,
                       json={**QUESTION, "overrides": {"critic_mode": "logprobs"}}).status_code == 400
    assert client.post("/v1/answer", headers=ADMIN, json={"question": ""}).status_code == 400


def test_stream_sends_sse_events_ending_in_done(api):
    client, _ = api
    with client.stream("POST", "/v1/answer/stream", json={**QUESTION, "mode": "sentence"},
                       headers=user()) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        text = r.read().decode()
    events = [line.removeprefix("event: ") for line in text.splitlines()
              if line.startswith("event: ")]
    assert events[0] == "segment" and events[-1] == "done"


def test_stream_rejects_forbidden_overrides_before_streaming(api):
    client, _ = api
    r = client.post("/v1/answer/stream", json={**QUESTION, "overrides": {"k": 3}}, headers=user())
    assert r.status_code == 403


def wait_for(client, job_id):
    for _ in range(100):
        job = client.get(f"/v1/jobs/{job_id}", headers=ADMIN).json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_upload_runs_an_ingestion_job(api, tmp_path):
    client, services = api
    r = client.post("/v1/documents", headers=ADMIN,
                    data={"tenant_id": "acme", "collection": "hr", "acl": "hr, execs"},
                    files=[("files", ("policy.md", b"# Notice\n30 days.", "text/markdown")),
                           ("files", ("bad.md", b"x", "text/markdown"))])
    assert r.status_code == 202
    job = wait_for(client, r.json()["job_id"])
    assert job["status"] == "failed"            # one of the two files failed
    good, bad = job["files"]
    assert (good["status"], good["doc_id"], good["added"]) == ("ready", "doc-policy", 3)
    assert bad["status"] == "failed" and bad["error"] == "cannot parse"
    path, tenant, collection, acl = services.pipeline.calls[0]
    assert path == tmp_path / "acme" / "hr" / "policy.md" and path.read_bytes().startswith(b"# Notice")
    assert (tenant, collection, acl) == ("acme", "hr", ["hr", "execs"])


def test_upload_validation_and_admin_only(api):
    client, _ = api
    md = [("files", ("a.md", b"x", "text/markdown"))]
    assert client.post("/v1/documents", files=md, headers=user()).status_code == 403
    assert client.post("/v1/documents", headers=ADMIN,
                       files=[("files", ("a.txt", b"x", "text/plain"))]).status_code == 400
    assert client.post("/v1/documents", headers=ADMIN, files=md,
                       data={"collection": "../escape"}).status_code == 400
    assert client.get("/v1/jobs/job_missing", headers=ADMIN).status_code == 404


def test_delete_document(api):
    client, services = api
    assert client.delete("/v1/documents/known", headers=ADMIN).status_code == 204
    assert client.delete("/v1/documents/missing", headers=ADMIN).status_code == 404
    assert client.delete("/v1/documents/known", headers=user()).status_code == 403
    assert services.deleted == ["known", "missing"]


def test_debug_trace_is_admin_only(api):
    client, _ = api
    r = client.post("/v1/answer", json={**QUESTION, "debug": True}, headers=ADMIN)
    assert r.status_code == 200 and r.json()["debug"]["steps"]
    assert client.post("/v1/answer", json=QUESTION, headers=ADMIN).json()["debug"] is None
    assert client.post("/v1/answer", json={**QUESTION, "debug": True},
                       headers=user()).status_code == 403
    assert client.post("/v1/answer/stream", json={**QUESTION, "debug": True},
                       headers=user()).status_code == 403
