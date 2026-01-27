"""
Setup script for the Multimodal RAG system.
Creates the required directories and configuration file for offline execution.
"""

from __future__ import annotations

import json
from pathlib import Path

from .enhanced_rag_pipeline import DEFAULT_CONFIG


def setup_directories() -> None:
    paths = DEFAULT_CONFIG["paths"]
    for directory in (
        ".data",
        paths["uploads"],
        paths["index_dir"],
        ".data/cache",
    ):
        Path(directory).mkdir(parents=True, exist_ok=True)


def write_config() -> Path:
    config_path = Path(".data/config.json")
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2))
    return config_path


def main() -> None:
    setup_directories()
    config_path = write_config()
    print("Multimodal RAG workspace prepared.")
    print(f"Configuration saved to {config_path.resolve()}")


if __name__ == "__main__":
    main()
