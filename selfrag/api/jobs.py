"""In-process ingestion jobs: uploads are ingested one job at a time in the background.

Job status lives in memory, so it is lost on restart (the ingested data is not).
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from selfrag.ingestion.pipeline import IngestionPipeline

log = logging.getLogger(__name__)


class FileResult(BaseModel):
    filename: str
    status: Literal["pending", "ready", "unchanged", "failed"] = "pending"
    doc_id: str | None = None
    added: int = 0
    removed: int = 0
    unchanged: int = 0
    error: str | None = None


class Job(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed"]   # failed: any file failed
    tenant_id: str
    collection: str
    acl: list[str]
    files: list[FileResult]
    created_at: datetime
    finished_at: datetime | None = None


class JobRunner:
    def __init__(self, pipeline: IngestionPipeline):
        self.pipeline = pipeline
        self.lock = asyncio.Lock()          # also held by document deletes
        self._jobs: dict[str, Job] = {}
        self._tasks: set[asyncio.Task] = set()

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def submit(self, paths: list[Path], *, tenant_id: str, collection: str,
               acl: list[str]) -> Job:
        job = Job(job_id=f"job_{uuid.uuid4().hex}", status="queued", tenant_id=tenant_id,
                  collection=collection, acl=acl,
                  files=[FileResult(filename=p.name) for p in paths],
                  created_at=datetime.now(timezone.utc))
        self._jobs[job.job_id] = job
        task = asyncio.create_task(self._run(job, paths))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return job

    async def _run(self, job: Job, paths: list[Path]) -> None:
        async with self.lock:
            job.status = "running"
            for result, path in zip(job.files, paths):
                try:
                    r = await self.pipeline.ingest_file(path, tenant_id=job.tenant_id,
                                                        collection=job.collection, acl=job.acl)
                    result.status, result.doc_id = r.status, r.doc_id
                    result.added, result.removed, result.unchanged = r.added, r.removed, r.unchanged
                except Exception as exc:
                    log.exception("ingest failed: %s (job %s)", path, job.job_id)
                    result.status, result.error = "failed", str(exc)
            job.status = "failed" if any(f.status == "failed" for f in job.files) else "succeeded"
            job.finished_at = datetime.now(timezone.utc)

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
