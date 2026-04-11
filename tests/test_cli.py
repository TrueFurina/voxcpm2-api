from __future__ import annotations

import types

from voxcpm2_api import cli


def test_cli_uses_asyncio_loop_on_macos(monkeypatch):
    calls = {}
    settings = types.SimpleNamespace(api_host="127.0.0.1", api_port=8000)

    monkeypatch.setattr(cli, "apply_torch_compat_patches", lambda: None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_app", lambda settings=None: object())
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))

    cli.main()

    assert calls["loop"] == "asyncio"


def test_cli_uses_auto_loop_off_macos(monkeypatch):
    calls = {}
    settings = types.SimpleNamespace(api_host="127.0.0.1", api_port=8000)

    monkeypatch.setattr(cli, "apply_torch_compat_patches", lambda: None)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "create_app", lambda settings=None: object())
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: calls.update(kwargs))

    cli.main()

    assert calls["loop"] == "auto"
