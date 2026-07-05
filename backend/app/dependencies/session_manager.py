from fastapi import HTTPException
import uuid


class encrypted_info:
    name: str
    stored_name: str
    id: str

    def __init__(self, id, stored_name, name):
        self.id = id
        self.stored_name = stored_name
        self.name = name


class session:
    datasets: list[encrypted_info]
    projects: list[encrypted_info]

    def __init__(self):
        self.datasets = []
        self.projects = []


sessions: dict[str, session] = {}


def delete_dataset(session_id, dataset_id):
    session = get_session(session_id)
    if session:
        session.datasets = [d for d in session.datasets if d.id != dataset_id]
    raise HTTPException(400, "Session not found")


def add_dataset(session_id, stored_name, name):
    session = get_session(session_id)
    if session:
        id = uuid.uuid4()
        session.datasets.append(
            encrypted_info(id=str(id), stored_name=str(stored_name), name=name)
        )
        return id
    raise HTTPException(400, "Session not found")


def add_project(session_id, stored_name, name):
    session = get_session(session_id)
    if session:
        id = uuid.uuid4()
        session.projects.append(
            encrypted_info(id=str(id), stored_name=str(stored_name), name=name)
        )
        return id
    raise HTTPException(400, "Session not found")


def get_stored_dataset(session_id, dataset_id):
    session = sessions.get(session_id, None)
    if not session:
        return None
    return next(
        (d.stored_name for d in session.datasets if str(d.id) == dataset_id), None
    )


def get_stored_project(session_id, project_id):
    session = sessions.get(session_id, None)
    if not session:
        return None
    return next(
        (d.stored_name for d in session.projects if str(d.id) == project_id), None
    )


def create_session():
    id = uuid.uuid4()
    sessions[str(id)] = session()
    return id


def get_session(id: str):
    return sessions.get(id, None)
