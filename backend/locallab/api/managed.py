from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from locallab.db import get_session
from locallab.models import ManagedServer, ModelRecord, RuntimeConfig
from locallab.runtimes.vllm_compat import ManagedVllmConfig
from locallab.schemas.managed import ManagedServerOut, RuntimeConfigIn, RuntimeConfigOut
from locallab.services.gpu_preflight import check_gpu

router = APIRouter(tags=["managed"])


@router.get("/runtime-configs", response_model=list[RuntimeConfigOut])
def list_runtime_configs(session: Session = Depends(get_session)):
    return session.scalars(select(RuntimeConfig).order_by(RuntimeConfig.name)).all()


@router.post("/runtime-configs", response_model=RuntimeConfigOut, status_code=201)
def create_runtime_config(body: RuntimeConfigIn, session: Session = Depends(get_session)):
    try:
        cfg = ManagedVllmConfig.model_validate(body.config)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False, include_context=False)) from e
    if body.registry_model_id is not None and session.get(ModelRecord, body.registry_model_id) is None:
        raise HTTPException(422, f"registry model {body.registry_model_id} not found")
    rec = RuntimeConfig(name=body.name.strip(), registry_model_id=body.registry_model_id,
                        config=cfg.model_dump(exclude={"allow_contention"}))  # never persist a contention override
    session.add(rec)
    try:
        session.flush()
    except IntegrityError as e:
        session.rollback()
        raise HTTPException(409, f"a runtime config named {body.name!r} already exists") from e
    return rec


@router.delete("/runtime-configs/{config_id}", status_code=204)
def delete_runtime_config(config_id: int, session: Session = Depends(get_session)):
    rec = session.get(RuntimeConfig, config_id)
    if rec is None:
        raise HTTPException(404, "runtime config not found")
    session.delete(rec)


@router.get("/managed-servers", response_model=list[ManagedServerOut])
def list_managed_servers(limit: int = 50, session: Session = Depends(get_session)):
    rows = session.scalars(select(ManagedServer).order_by(ManagedServer.id.desc()).limit(min(limit, 200))).all()
    return [ManagedServerOut.model_validate(r).model_copy(update={"log_tail": None}) for r in rows]  # list omits logs


@router.get("/managed-servers/{server_id}", response_model=ManagedServerOut)
def get_managed_server(server_id: int, session: Session = Depends(get_session)):
    row = session.get(ManagedServer, server_id)
    if row is None:
        raise HTTPException(404, "managed server not found")
    return row


@router.get("/gpu/preflight")
def gpu_preflight(gpu_index: int = 0, utilization: float = 0.3):
    """What a launch would see right now (free memory, other compute processes, contention)."""
    return check_gpu(gpu_index, min(max(utilization, 0.05), 0.95), allow_contention=False).as_dict()
