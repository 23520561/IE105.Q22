from fastapi import APIRouter, Cookie

import app.dataset_project.service as Service
from app.dataset_project.schema import ProjectRequest
from app.dependencies.session_manager import add_project

router = APIRouter(
    prefix="/project",
    tags=["dataset, project"],
    responses={404: {"description": "Not found"}},
)


@router.get("")
def get_projects(x_session_id: str = Cookie(...)):
    return Service.get_projects(x_session_id)


@router.post("")
def create_projects(req: ProjectRequest, x_session_id: str = Cookie(...)):
    stored_name = Service.create_project(
        dataset_id=req.dataset_id, x_session_id=x_session_id
    )
    return add_project(session_id=x_session_id, stored_name=stored_name, name=req.name)
