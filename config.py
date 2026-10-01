"""读取本地模型配置。"""
import json
from pathlib import Path


def load_config(model_path: str | Path) -> dict:
    """统一读取本地模型配置，供引擎、加载器和对照脚本复用。"""
    with (Path(model_path).expanduser() / "config.json").open() as f:
        config = json.load(f)
    return config

