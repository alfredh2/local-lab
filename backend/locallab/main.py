import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from locallab import __version__
from locallab.api import system
from locallab.config import get_settings
from locallab.db import init_db
from locallab.logging import configure_logging
from locallab.services.gpu import GpuHistory

log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        init_db(settings)
        app.state.gpu_history = GpuHistory()
        app.state.gpu_history.start()
        log.info("LocalLab API %s started (db=%s)", __version__, settings.database_url)
        yield
        app.state.gpu_history.stop()

    app = FastAPI(title="LocalLab", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
    )
    app.include_router(system.router, prefix="/api")
    return app


app = create_app()
