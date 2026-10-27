from locallab.evals.base import EvalSuite
from locallab.evals.datasets.coding import CodingSuite
from locallab.evals.datasets.retrieval import RetrievalSuite
from locallab.evals.datasets.structured import StructuredSuite

SUITES: dict[str, EvalSuite] = {s.name: s for s in (StructuredSuite(), CodingSuite(), RetrievalSuite())}


def get_suite(name: str) -> EvalSuite:
    try:
        return SUITES[name]
    except KeyError:
        raise ValueError(f"unknown eval suite {name!r}; available: {sorted(SUITES)}") from None
