"""
Backtest queue endpoints powering historical replay tooling.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status

from schemas.backtest import BacktestEnqueueRequest, BacktestJob, BacktestRunOptions
from services.backtest_service import BacktestService

router = APIRouter()
service = BacktestService()


@router.get("/queue", response_model=List[BacktestJob])
async def list_backtest_queue() -> List[BacktestJob]:
    """Return current backtest queue ordered by creation time."""
    return service.list_jobs()


@router.post("/queue", response_model=BacktestJob, status_code=status.HTTP_201_CREATED)
async def enqueue_backtest_job(request: BacktestEnqueueRequest) -> BacktestJob:
    """Create a new backtest job and return its metadata."""
    try:
        return service.enqueue(request)
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to enqueue backtest job: {exc}") from exc


@router.get("/queue/{job_id}", response_model=BacktestJob)
async def get_backtest_job(job_id: str) -> BacktestJob:
    """Return a single backtest job by ID."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")
    return job


@router.delete("/queue/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_backtest_job(job_id: str) -> None:
    """Remove a backtest job."""
    removed = service.remove_job(job_id)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Backtest job {job_id} not found")


@router.post("/queue/{job_id}/run", response_model=BacktestJob)
async def run_backtest_job(job_id: str, options: Optional[BacktestRunOptions] = None) -> BacktestJob:
    """Execute an existing backtest job immediately."""
    try:
        return service.run_job(job_id, options)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Failed to run backtest job: {exc}") from exc
