from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from locallab.config import get_settings
from locallab.db import get_session
from locallab.models import ModelRecord
from locallab.schemas.models import ManualModelIn, ModelOut, ScanResult
from locallab.services import registry

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelOut])
def list_models(session: Session = Depends(get_session)):
    return session.scalars(select(ModelRecord).order_by(ModelRecord.name)).all()


@router.post("/scan", response_model=ScanResult)
def scan(session: Session = Depends(get_session)):
    return registry.sync_registry(session, get_settings())


@router.post("", response_model=ModelOut, status_code=201)
def add(body: ManualModelIn, session: Session = Depends(get_session)):
    try:
        return registry.add_manual(session, body.name, body.path)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


@router.delete("/{model_id}", status_code=204)
def delete(model_id: int, session: Session = Depends(get_session)):
    rec = session.get(ModelRecord, model_id)
    if rec is None:
        raise HTTPException(404, "Model not found")
    session.delete(rec)
