from __future__ import annotations

from pathlib import Path


def case_relative_path(path: Path, case_dir: Path) -> str:
    """Return `path` relative to the slide case directory."""
    return path.resolve().relative_to(case_dir.resolve()).as_posix()


def resolve_case_path(
    value: str | Path,
    case_dir: Path,
) -> Path:
    """Resolve a case-relative file reference."""
    path = Path(value)
    if path.is_absolute():
        return path
    return case_dir.resolve() / path
