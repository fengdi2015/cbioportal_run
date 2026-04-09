from __future__ import annotations

import sys
import os
from uuid import uuid4
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

TEST_TMP_ROOT = ROOT / "downloads" / "tmp-tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture
def tmp_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if os.name != "nt":
        return tmp_path_factory.mktemp("case")
    path = TEST_TMP_ROOT / f"case-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path
