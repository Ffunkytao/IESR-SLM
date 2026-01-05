from pathlib import Path
from typing import Dict
import os

# Get the directory of this file, then go up one level to find templates
_current_dir = Path(__file__).parent
_project_root = _current_dir.parent.parent
TEMPLATE_DIR = _project_root / "alphasql_rstar" / "templates"

# Fallback to relative path if absolute path doesn't work
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = Path(__file__).parent.parent / "templates"

TEMPLATE_DICT = {}

for template_file in TEMPLATE_DIR.glob("*.txt"):
    with open(template_file, "r") as f:
        TEMPLATE_DICT[template_file.stem] = f.read()

def get_prompt(template_name: str, template_args: Dict[str, str]) -> str:
    template = TEMPLATE_DICT[template_name]
    return template.format(**template_args)
