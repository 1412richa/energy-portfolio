from pathlib import Path


def find_project_root(name: str = "01-price-forecasting") -> Path:
    """Locate the project folder from this file's location, so it works from any working directory."""
    here = Path(__file__).resolve().parent
    for folder in [here, *here.parents]:
        if folder.name == name:
            return folder
        if (folder / name).is_dir():
            return folder / name
    raise FileNotFoundError(f"Could not find '{name}' from {here}")
