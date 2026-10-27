from locallab.experiments.base import Experiment
from locallab.experiments.diagnostic import DiagnosticExperiment

_TYPES: dict[str, Experiment] = {e.type_name: e for e in (DiagnosticExperiment(),)}


def get_experiment(type_name: str) -> Experiment:
    try:
        return _TYPES[type_name]
    except KeyError:
        raise ValueError(f"Unknown experiment type: {type_name!r}") from None


def list_experiments() -> list[Experiment]:
    return list(_TYPES.values())
