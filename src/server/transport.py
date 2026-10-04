import hmac
import os
import secrets
import tempfile
from pathlib import Path

import requests

TOKEN_ENV = "SKINTOKENS_SERVER_TOKEN"
PAYLOAD_DIR_ENV = "SKINTOKENS_PAYLOAD_DIR"


def prepare_server_session():
    os.environ[TOKEN_ENV] = secrets.token_urlsafe(32)
    os.environ[PAYLOAD_DIR_ENV] = tempfile.mkdtemp(prefix="skintokens_session_")


def server_token():
    token = os.environ.get(TOKEN_ENV, "")
    if len(token) < 32:
        raise RuntimeError("Start the Blender server through SkinTokens, or set SKINTOKENS_SERVER_TOKEN and SKINTOKENS_PAYLOAD_DIR for manual use.")
    return token


def authenticated(authorization):
    return hmac.compare_digest(authorization or "", "Bearer " + server_token())


def payload_directory():
    directory = os.environ.get(PAYLOAD_DIR_ENV)
    if not directory or not Path(directory).is_dir():
        raise RuntimeError("SkinTokens session directory is missing")
    return Path(directory).resolve()


def resolve_payload_path(name):
    if not isinstance(name, str) or not name.startswith("skintokens_") or not name.endswith(".pt"):
        raise ValueError("Invalid temporary payload name")
    if "/" in name or "\\" in name or ":" in name or name in (".", ".."):
        raise ValueError("Payload must be a session filename")
    directory = payload_directory()
    path = directory / name
    if path.is_symlink() or path.resolve().parent != directory or not path.is_file():
        raise ValueError("Payload is outside the SkinTokens session")
    return path


def server_request(method, url, **kwargs):
    headers = dict(kwargs.pop("headers", {}))
    headers["Authorization"] = "Bearer " + server_token()
    headers["X-SkinTokens-Protocol"] = "1"
    # Loopback traffic must never send the session secret through an HTTP proxy.
    with requests.Session() as session:
        session.trust_env = False
        result = session.request(method, url, headers=headers, **kwargs)
    result.raise_for_status()
    if result.headers.get("X-SkinTokens-Protocol") != "1":
        raise RuntimeError("The local port is not running the secured SkinTokens server")
    return result
