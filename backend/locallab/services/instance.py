import secrets

from locallab.config import Settings


def get_instance_id(settings: Settings) -> str:
    """Stable per-install ID; stamped on every managed container so ownership is unambiguous."""
    path = settings.instance_id_path
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        iid = secrets.token_hex(8)
        path.write_text(iid)
        return iid
