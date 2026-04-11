from __future__ import annotations

import uvicorn

from voxcpm2_api.config import get_settings
from voxcpm2_api.main import create_app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.api_host,
        port=settings.api_port,
    )
