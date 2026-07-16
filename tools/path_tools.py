from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path):
    """Resolve relative CLI paths from the AdaCLIP repository root."""
    if path is None or str(path).strip() == "":
        return path

    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return str(resolved.resolve())


def resolve_arg_paths(args, names):
    for name in names:
        if hasattr(args, name):
            setattr(args, name, resolve_repo_path(getattr(args, name)))
    return args
