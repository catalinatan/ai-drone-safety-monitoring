from pathlib import Path


def find_project_root(start: Path | None = None):
    """Based on config files, derive root path of project folder.

    Args:
        start (Path | None, optional): Starting path to search from.

    Returns:
        Path: Root path of the project
    """
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    # Fallback: topmost directory we reached
    return start.parents[-1]
