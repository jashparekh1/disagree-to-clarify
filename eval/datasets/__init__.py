"""Unified dataset loading for D2C evaluation."""

from eval.datasets.base import AmbiguousQuery


def load_dataset(name: str, data_dir: str = "data") -> list[AmbiguousQuery]:
    """Load any supported dataset by name."""
    if name == "clamber":
        from eval.datasets.load_clamber import load_clamber
        return load_clamber(f"{data_dir}/clamber")
    elif name == "qulac":
        from eval.datasets.load_qulac import load_qulac
        return load_qulac(f"{data_dir}/qulac")
    elif name == "clariq":
        from eval.datasets.load_clariq import load_clariq
        return load_clariq(f"{data_dir}/clariq")
    elif name == "abgcoqa":
        from eval.datasets.load_abgcoqa import load_abgcoqa
        return load_abgcoqa(f"{data_dir}/abgcoqa")
    else:
        raise ValueError(f"Unknown dataset: {name}. Choose from: clamber, qulac, clariq, abgcoqa")


def load_all_datasets(data_dir: str = "data") -> dict[str, list[AmbiguousQuery]]:
    """Load all datasets. Returns dict mapping name -> list."""
    return {name: load_dataset(name, data_dir) for name in ["clamber", "qulac", "clariq", "abgcoqa"]}
