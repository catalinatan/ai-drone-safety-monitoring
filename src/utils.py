from pathlib import Path

def find_project_root(start: Path | None = None):
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback: topmost directory we reached
    return start.parents[-1]