"""Centralized MLflow logging for ZT-Pipeline federated learning experiments.

All MLflow API calls go through :class:`ExperimentTracker`.  When tracking
is disabled (``--no-mlflow``), substitute :class:`NullTracker` – same
interface, zero side-effects.

Usage
-----
>>> tracker = ExperimentTracker("./mlruns", "ZT-Pipeline")
>>> tracker.start_run("full_zt", config_dict={...})
>>> tracker.log_metrics({"global_accuracy": 0.87}, step=5)
>>> tracker.end_run()
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Real tracker (wraps MLflow)
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Encapsulates all MLflow tracking logic.

    Parameters
    ----------
    tracking_uri : str
        Local or remote MLflow tracking server URI.
        Default ``"./mlruns"`` stores everything on disk.
    experiment_name : str
        Name of the MLflow experiment.  Created automatically if it does
        not already exist.
    """

    def __init__(
        self,
        tracking_uri: str = "./mlruns",
        experiment_name: str = "ZT-Pipeline",
    ) -> None:
        import mlflow  # lazily imported so NullTracker works without mlflow
        self._mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    # -- Experiment management ---------------------------------------------

    def set_experiment(self, name: str) -> None:
        """Switch the active MLflow experiment."""
        self._mlflow.set_experiment(name)

    # -- Run lifecycle -----------------------------------------------------

    def start_run(
        self,
        run_name: str,
        config_dict: Optional[Dict[str, Any]] = None,
        nested: bool = False,
    ) -> str:
        """Start a new MLflow run (optionally nested under the active run).

        Returns the *run_id* string.
        """
        run = self._mlflow.start_run(run_name=run_name, nested=nested)
        if config_dict:
            self.log_params(config_dict)
        return run.info.run_id

    def end_run(self) -> None:
        """End the current active MLflow run."""
        self._mlflow.end_run()

    # -- Parameter logging -------------------------------------------------

    def log_params(self, params_dict: Dict[str, Any]) -> None:
        """Log a flat dict of parameters.

        Non-scalar values (lists, dicts, None) are converted to strings
        so MLflow can store them.
        """
        safe: Dict[str, Any] = {}
        for k, v in params_dict.items():
            if v is None:
                safe[k] = "None"
            elif isinstance(v, (list, dict)):
                safe[k] = str(v)
            else:
                safe[k] = v
        try:
            self._mlflow.log_params(safe)
        except Exception as exc:
            # Duplicate-param or overflow → warn but don't crash the run
            print(f"[mlflow] Warning logging params: {exc}")

    # -- Metric logging ----------------------------------------------------

    def log_metrics(
        self,
        metrics_dict: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """Log numeric metrics.  Non-numeric / None values are skipped."""
        for k, v in metrics_dict.items():
            if v is not None and isinstance(v, (int, float)):
                self._mlflow.log_metric(k, float(v), step=step)

    # -- Artifact logging --------------------------------------------------

    def log_artifact(self, file_path: Union[str, Path]) -> None:
        """Log a local file as an MLflow artifact."""
        p = Path(file_path)
        if p.exists():
            self._mlflow.log_artifact(str(p))

    def log_model(self, model: Any, artifact_path: str = "final_model") -> None:
        """Log a PyTorch model via ``mlflow.pytorch.log_model``."""
        try:
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as exc:
            print(f"[mlflow] Could not log PyTorch model: {exc}")

    # -- System & reproducibility ------------------------------------------

    def log_system_info(self) -> None:
        """Log Python / Torch / CUDA versions as run parameters."""
        info: Dict[str, str] = {"python_version": sys.version.split()[0]}
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch_version"] = "not_installed"
        self.log_params(info)

    def log_reproducibility_info(self) -> None:
        """Log git commit hash, Docker image tags, and *pip freeze* output."""
        # Git commit hash
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._mlflow.log_param(
                    "git_commit", result.stdout.strip()[:40]
                )
        except Exception:
            pass

        # Docker image tags
        try:
            self._mlflow.log_param("zt_docker_image", "zt-fl-experiment")
            self._mlflow.log_param(
                "baseline_docker_image", "baseline-fl-experiment"
            )
        except Exception:
            pass

        # pip freeze → artifact
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                tmp = Path(tempfile.mktemp(suffix="_pip_freeze.txt"))
                tmp.write_text(result.stdout, encoding="utf-8")
                self._mlflow.log_artifact(str(tmp))
                tmp.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Null-object tracker (no-op when MLflow is disabled)
# ---------------------------------------------------------------------------

class NullTracker:
    """Drop-in replacement for :class:`ExperimentTracker` that does nothing.

    Used when the ``--no-mlflow`` flag is set so the rest of the code
    can call the same methods unconditionally.

    Every method exactly mirrors the :class:`ExperimentTracker` public
    interface so callers need no ``if mlflow_enabled`` guards.
    """

    def set_experiment(self, *a: Any, **kw: Any) -> None: ...
    def start_run(self, *a: Any, **kw: Any) -> str: return ""
    def end_run(self, *a: Any, **kw: Any) -> None: ...
    def log_params(self, *a: Any, **kw: Any) -> None: ...
    def log_metrics(self, *a: Any, **kw: Any) -> None: ...
    def log_artifact(self, *a: Any, **kw: Any) -> None: ...
    def log_model(self, *a: Any, **kw: Any) -> None: ...
    def log_system_info(self, *a: Any, **kw: Any) -> None: ...
    def log_reproducibility_info(self, *a: Any, **kw: Any) -> None: ...
    def log_params(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
    def log_metrics(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
    def log_artifact(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
    def log_model(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
    def log_system_info(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
    def log_reproducibility_info(self, *a: Any, **kw: Any) -> None: ...  # noqa: E704
