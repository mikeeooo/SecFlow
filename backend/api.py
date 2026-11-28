"""FastAPI application and route registration.

Run a single process for the intended local workflow:

    uv run uvicorn backend.api:app --host 127.0.0.1 --port 8001
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    AccountPage,
    AssessmentDetail,
    Investigation,
    LiveFeed,
    ReviewHistoryPage,
    ReviewRequest,
    SimulationRequest,
    StreamModeRequest,
    StreamStatus,
    TransactionPage,
    TransactionQuery,
)
from backend.service import ConflictError, MiaLimitError, NotFoundError, SecFlowService
from backend.stream import StreamController
from config.settings import ROOT, Settings, load_settings
from datagen.records import format_ts

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("secflow.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = SecFlowService(settings)
        stream = StreamController(service, settings.stream_interval_seconds, settings.batch_interval_seconds)
        app.state.service = service
        app.state.stream = stream
        if settings.autostart_stream:
            stream.start()
            log.info("Simulated transfer stream started with the server")
        try:
            yield
        finally:
            await stream.shutdown()

    app = FastAPI(
        title="SecFlow API",
        version="1.0.0",
        description="Explainable payment-risk analysis. Scores are anomaly-risk indices, not fraud probabilities.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    def error(status: int, code: str, message: str, details: Any = None) -> JSONResponse:
        return JSONResponse(
            status_code=status, content={"error": {"code": code, "message": message, "details": details}}
        )

    @app.exception_handler(MiaLimitError)
    async def mia_limit(_: Request, exc: MiaLimitError) -> JSONResponse:
        return error(422, "validation_error", str(exc), {"limit": "mia_per_transaction"})

    @app.exception_handler(NotFoundError)
    async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return error(404, "not_found", f"Not found: {exc.args[0] if exc.args else ''}")

    @app.exception_handler(ConflictError)
    async def conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return error(409, "conflict", str(exc))

    @app.exception_handler(RequestValidationError)
    async def invalid(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            {"field": ".".join(str(p) for p in e["loc"][1:]), "message": e["msg"]} for e in exc.errors()
        ]
        return error(422, "validation_error", "Some fields need attention.", details)

    @app.exception_handler(HTTPException)
    async def http_error(_: Request, exc: HTTPException) -> JSONResponse:
        return error(exc.status_code, "http_error", str(exc.detail))

    def svc(request: Request) -> SecFlowService:
        service: SecFlowService = request.app.state.service
        return service

    def stream_of(request: Request) -> StreamController:
        stream: StreamController = request.app.state.stream
        return stream

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        return svc(request).health(stream_of(request).status())

    @app.get("/api/overview")
    def overview(request: Request) -> dict[str, Any]:
        data = svc(request).overview()
        data["stream"] = stream_of(request).status()
        return data

    @app.get("/api/transactions", response_model=TransactionPage)
    def transactions(request: Request, filters: Annotated[TransactionQuery, Query()]) -> TransactionPage:
        return svc(request).transactions(filters)

    @app.get("/api/accounts", response_model=AccountPage)
    def accounts(
        request: Request,
        q: str | None = None,
        account_type: Annotated[str | None, Query(pattern="^(personal|business)$")] = None,
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=1000)] = 30,
    ) -> AccountPage:
        return svc(request).accounts_page(q, page, page_size, account_type)

    @app.get("/api/accounts/{account_id}")
    def account(request: Request, account_id: str) -> dict[str, Any]:
        return svc(request).account_detail(account_id)

    @app.get("/api/parties")
    def parties(
        request: Request, q: str = "", limit: Annotated[int, Query(ge=1, le=50)] = 20
    ) -> list[dict[str, Any]]:
        return svc(request).parties(q, limit)

    @app.get("/api/simulator/options")
    def simulator_options(request: Request, sender_id: str | None = None) -> dict[str, Any]:
        return svc(request).simulator_options(sender_id)

    @app.post("/api/simulations", response_model=AssessmentDetail)
    def simulate(request: Request, payload: SimulationRequest, response: Response) -> AssessmentDetail:
        detail, replay = svc(request).simulate(payload, "manual")
        response.status_code = 200 if replay else 201
        return detail

    @app.get("/api/live-transfers", response_model=LiveFeed)
    def live_transfers(
        request: Request,
        after: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> LiveFeed:
        service = svc(request)
        items = service.live_feed(after, limit)
        next_cursor = max([after] + [i.cursor for i in items])
        return LiveFeed(
            items=items,
            next_cursor=next_cursor,
            stream=StreamStatus(**stream_of(request).status()),
            simulation_time=format_ts(service.simulation_now()),
        )

    @app.get("/api/simulation-stream", response_model=StreamStatus)
    def stream_status(request: Request) -> StreamStatus:
        return StreamStatus(**stream_of(request).status())

    @app.post("/api/simulation-stream/start", response_model=StreamStatus)
    async def stream_start(request: Request) -> StreamStatus:
        return StreamStatus(**stream_of(request).start())

    @app.post("/api/simulation-stream/stop", response_model=StreamStatus)
    async def stream_stop(request: Request) -> StreamStatus:
        return StreamStatus(**stream_of(request).stop())

    @app.post("/api/simulation-stream/mode", response_model=StreamStatus)
    async def stream_mode(request: Request, payload: StreamModeRequest) -> StreamStatus:
        return StreamStatus(**stream_of(request).set_mode(payload.mode))

    @app.get("/api/investigations/{request_id}", response_model=Investigation)
    def investigation(request: Request, request_id: str) -> Investigation:
        return svc(request).investigation(request_id)

    @app.post("/api/investigations/{request_id}/reviews", response_model=Investigation, status_code=201)
    def review(request: Request, request_id: str, payload: ReviewRequest) -> Investigation:
        return svc(request).review(request_id, payload)

    @app.get("/api/reviews", response_model=ReviewHistoryPage)
    def review_history(
        request: Request,
        action: Annotated[str | None, Query(pattern="^(approve|reject)$")] = None,
        q: Annotated[str | None, Query(max_length=100)] = None,
        sort: Annotated[str, Query(pattern="^(newest|oldest)$")] = "newest",
        page: Annotated[int, Query(ge=1)] = 1,
        page_size: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> ReviewHistoryPage:
        return svc(request).review_history(action, q, page, page_size, sort)

    @app.get("/api/algorithms")
    def algorithms(request: Request) -> dict[str, Any]:
        return svc(request).algorithms()

    @app.get("/api/dashboard")
    def monitoring_dashboard(
        request: Request, window: Annotated[str, Query(pattern="^(history|live)$")] = "history"
    ) -> dict[str, Any]:
        return svc(request).dashboard(window)

    @app.get("/api/algorithms/cluster-graph")
    def cluster_graph(
        request: Request, anomalies: Annotated[int, Query(ge=0, le=500)] = 150
    ) -> dict[str, Any]:
        return svc(request).cluster_graph(anomalies)

    @app.get("/api/evaluation")
    def evaluation(request: Request) -> dict[str, Any]:
        data = svc(request).evaluation
        if data is None:
            raise HTTPException(404, "No evaluation report yet. Run `python -m backend.cli evaluate`.")
        return data

    _mount_frontend(app)
    return app


def _mount_frontend(app: FastAPI, dist: Path = ROOT / "frontend" / "dist") -> None:
    """Serve the production frontend build from the same process when it exists.

    Files in the build are served as they are. Any other path gets the single-page app, except a path
    with a file extension, which gets 404, so a missing robots.txt or image is not answered with HTML.
    """
    if not (dist / "index.html").exists():
        return
    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        if path.startswith("api/"):
            raise HTTPException(404, "Unknown API route")
        candidate = (dist / path).resolve()
        if path and candidate.is_file() and Path(dist.resolve()) in candidate.parents:
            return FileResponse(candidate)
        if Path(path).suffix:
            raise HTTPException(404, "File not found")
        return FileResponse(dist / "index.html")


app = create_app()
