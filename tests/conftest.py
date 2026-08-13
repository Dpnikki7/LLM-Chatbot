import os
import sys
import tempfile

os.environ["USE_HF"] = "false"
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db  # noqa: E402

_tmp = tempfile.mkdtemp()
db.set_db_path(os.path.join(_tmp, "test.db"))

import app as app_module  # noqa: E402

client = app_module.app  # module-level import is replaced by a fixture below

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(app_module.app)