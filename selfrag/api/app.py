"""FastAPI service over the Self-RAG pipeline (API design in the software design plan).

    python -m selfrag.api                                   # or:
    uvicorn selfrag.api.app:create_app --factory
"""
import asyncio
import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Awaitable, BinaryIO, Callable

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from selfrag.api.auth import Principal, authenticate, require_admin
from selfrag.api.jobs import Job, JobRunner
from selfrag.api.schemas import AnswerRequest
from selfrag.api.services import Services, build_services
from selfrag.api.settings import ApiSettings
from selfrag.api.sse import encode
from selfrag.config import Settings
from selfrag.config.overrides import OverridesNotAllowed
from selfrag.ingestion.parsers import PARSERS
from selfrag.models import AnswerResponse, Filter
from selfrag.orchestrator.graph import InvalidOverride, SelfRAG

log = logging.getLogger(__name__)

NAME_PATTERN = r"^[A-Za-z0-9_.-]{1,64}$"       # tenant and collection names (also folder names)

ServicesFactory = Callable[[], Awaitable[Services]]


async def default_services() -> Services:
    return await build_services(Settings())


def get_services(request: Request) -> Services:
    return request.app.state.services


def get_jobs(request: Request) -> JobRunner:
    return request.app.state.jobs


def scope(req: AnswerRequest, who: Principal, services: Services) -> tuple[SelfRAG, Filter]:
    """The engine for the requested mode and the retrieval filter the caller may use."""
    f = req.filters
    if who.is_admin:
        filters = Filter(tenant_id=f.tenant_id or "default", collection=f.collection,
                         acl_groups=tuple(f.acl_groups or ()))
    elif f.tenant_id is not None or f.acl_groups is not None:
        raise HTTPException(status.HTTP_403_FORBIDDEN,
                            "tenant_id and acl_groups come from your token")
    else:
        filters = Filter(tenant_id=who.tenant_id, collection=f.collection,
                         acl_groups=who.acl_groups)
    return services.rags[req.mode or services.default_mode], filters


def http_error(exc: Exception) -> HTTPException:
    """Map an engine failure to the status codes in the design plan."""
    if isinstance(exc, OverridesNotAllowed):
        return HTTPException(status.HTTP_403_FORBIDDEN, str(exc))
    if isinstance(exc, (InvalidOverride, ValidationError)):
        return HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))
    log.exception("answer failed")
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                         "Answer generation failed: a model or store is unavailable.")


def save_upload(src: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy to a temp file, then swap it in, so a running ingest never reads half a file."""
    part = dest.with_name(dest.name + ".part")
    written = 0
    try:
        with part.open("wb") as out:
            while block := src.read(1 << 20):
                written += len(block)
                if written > max_bytes:
                    raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                        f"{dest.name} is larger than {max_bytes >> 20} MB")
                out.write(block)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)


def create_app(services_factory: ServicesFactory = default_services,
               api_settings: ApiSettings | None = None) -> FastAPI:
    api_settings = api_settings or ApiSettings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        services = await services_factory()         # loads the local models once
        jobs = JobRunner(services.pipeline)
        app.state.services, app.state.jobs = services, jobs
        try:
            yield
        finally:
            await jobs.close()
            await services.close()

    app = FastAPI(title="Self-RAG Service", version="0.1.0", lifespan=lifespan)
    app.state.api_settings = api_settings

    @app.exception_handler(RequestValidationError)
    async def bad_request(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST,
                            content={"detail": jsonable_encoder(exc.errors())})

    # ------------------------------------------------------------ health

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(services: Services = Depends(get_services)):
        checks = await services.ready()
        ok = all(checks.values())
        return JSONResponse(status_code=200 if ok else 503,
                            content={"status": "ready" if ok else "unavailable", "checks": checks})

    # ------------------------------------------------------------ answers

    @app.post("/v1/answer", response_model=AnswerResponse)
    async def answer(req: AnswerRequest, who: Principal = Depends(authenticate),
                     services: Services = Depends(get_services)):
        rag, filters = scope(req, who, services)
        try:
            return await rag.respond(req.question, filters, req.overrides, is_admin=who.is_admin,
                                     debug_trace=req.debug)
        except Exception as exc:
            raise http_error(exc) from exc

    @app.post("/v1/answer/stream")
    async def answer_stream(req: AnswerRequest, who: Principal = Depends(authenticate),
                            services: Services = Depends(get_services)):
        rag, filters = scope(req, who, services)
        try:            # reject bad overrides with a plain error before any event is sent
            rag.check_debug(req.debug, who.is_admin)
            rag.resolve_config(req.overrides, is_admin=who.is_admin)
        except Exception as exc:
            raise http_error(exc) from exc

        async def body():   # the whole stream runs in one task: usage tracking needs that
            async for event in rag.stream(req.question, filters, req.overrides,
                                          is_admin=who.is_admin, debug_trace=req.debug):
                yield encode(event)

        return StreamingResponse(body(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------ documents (admin)

    @app.post("/v1/documents", status_code=status.HTTP_202_ACCEPTED, response_model=Job)
    async def upload_documents(files: list[UploadFile] = File(...),
                               tenant_id: str = Form("default", pattern=NAME_PATTERN),
                               collection: str = Form("default", pattern=NAME_PATTERN),
                               acl: str = Form("", description="comma-separated ACL groups"),
                               _: Principal = Depends(require_admin),
                               jobs: JobRunner = Depends(get_jobs)):
        names = [Path(f.filename or "").name for f in files]
        for name in names:
            if Path(name).suffix.lower() not in PARSERS:
                raise HTTPException(status.HTTP_400_BAD_REQUEST,
                                    f"unsupported file {name!r} (supported: {', '.join(PARSERS)})")
        if len(set(names)) != len(names):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "duplicate file names")

        # A stable path per (tenant, collection, name) keeps the doc_id stable, so a
        # re-upload updates the document incrementally instead of adding a copy.
        folder = api_settings.upload_dir / tenant_id / collection
        folder.mkdir(parents=True, exist_ok=True)
        paths = [folder / name for name in names]
        for f, path in zip(files, paths):
            await asyncio.to_thread(save_upload, f.file, path, api_settings.max_upload_mb << 20)
        return jobs.submit(paths, tenant_id=tenant_id, collection=collection,
                           acl=[a.strip() for a in acl.split(",") if a.strip()])

    @app.get("/v1/jobs/{job_id}", response_model=Job)
    async def get_job(job_id: str, _: Principal = Depends(require_admin),
                      jobs: JobRunner = Depends(get_jobs)):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
        return job

    @app.delete("/v1/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
    async def delete_document(doc_id: str, _: Principal = Depends(require_admin),
                              services: Services = Depends(get_services),
                              jobs: JobRunner = Depends(get_jobs)):
        async with jobs.lock:               # never race an ingest of the same document
            found = await services.delete_document(doc_id)
        if not found:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # OpenAPI 3.1 marks uploads with contentMediaType only; Swagger UI needs
    # format: binary to render a file picker instead of a text box.
    default_openapi = app.openapi

    def openapi_with_file_pickers() -> dict:
        if app.openapi_schema is None:
            schema = default_openapi()
            for model in schema.get("components", {}).get("schemas", {}).values():
                for prop in model.get("properties", {}).values():
                    for s in (prop, prop.get("items", {})):
                        if s.get("contentMediaType") == "application/octet-stream":
                            s["format"] = "binary"
        return app.openapi_schema

    app.openapi = openapi_with_file_pickers
    return app
