class ExpectedFailure(Exception):
    """A failure with a user-facing message (bad config, server crash, ...): the job records
    `str(e)` as its error instead of a traceback."""


class ServerCrashed(ExpectedFailure):
    """The inference server died or vanished while a benchmark was running."""


class WorkerShutdown(Exception):
    """The worker process is stopping; the job cannot be completed."""
