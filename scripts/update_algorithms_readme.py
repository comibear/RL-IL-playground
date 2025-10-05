#!/usr/bin/env python3
"""Update README.md with a tree view of the algorithms directory."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER_START = "<!-- algorithms-tree start -->"
MARKER_END = "<!-- algorithms-tree end -->"
CODE_FENCE = "```"


def build_tree(root: Path) -> str:
    if not root.exists():
        raise FileNotFoundError(f"Algorithms directory not found: {root}")

    display_root = root.relative_to(root.parents[1]) if len(root.parents) > 1 else root
    lines: list[str] = [f"{display_root}/"]

    def iter_dir(path: Path, prefix: str = "") -> None:
        entries = [
            entry
            for entry in sorted(
                path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())
            )
            if not entry.name.startswith(".") and entry.name != "__pycache__"
        ]
        for index, entry in enumerate(entries):
            connector = "└── " if index == len(entries) - 1 else "├── "
            name = entry.name + ("/" if entry.is_dir() else "")
            lines.append(f"{prefix}{connector}{name}")
            if entry.is_dir():
                child_prefix = prefix + ("    " if index == len(entries) - 1 else "│   ")
                iter_dir(entry, child_prefix)

    iter_dir(root)
    return "\n".join(lines)


def update_readme(readme_path: Path, tree_text: str) -> bool:
    content = readme_path.read_text(encoding="utf-8")
    section = (
        f"{MARKER_START}\n{CODE_FENCE}\n{tree_text}\n{CODE_FENCE}\n{MARKER_END}\n"
    )

    if MARKER_START in content and MARKER_END in content:
        start_index = content.index(MARKER_START)
        end_index = content.index(MARKER_END) + len(MARKER_END)
        new_content = content[:start_index] + section + content[end_index:]
    else:
        suffix = "\n" if content.endswith("\n") else "\n\n"
        heading = "## Implemented algorithms\n"
        new_content = content + suffix + heading + section

    if not new_content.endswith("\n"):
        new_content += "\n"

    if new_content == content:
        return False

    readme_path.write_text(new_content, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the repository root",
    )
    args = parser.parse_args(argv)

    project_root: Path = args.project_root
    readme_path = project_root / "README.md"
    algorithms_dir = project_root / "RL" / "algorithms"

    tree_text = build_tree(algorithms_dir)
    update_readme(readme_path, tree_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
