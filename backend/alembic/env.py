from alembic import context
from sqlalchemy import pool

import locallab.models  # noqa: F401  (register tables)
from locallab.config import get_settings
from locallab.db import Base, make_engine

target_metadata = Base.metadata


def run() -> None:
    url = context.config.get_main_option("sqlalchemy.url") or get_settings().database_url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:///:memory:"):
        from pathlib import Path

        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = make_engine(url)
    if context.is_offline_mode():
        context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
        with context.begin_transaction():
            context.run_migrations()
        return
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, render_as_batch=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


run()
