"""Parliament of Owls - Query multiple LLMs in parallel."""

import logging

# The single source of truth for the version; pyproject.toml reads it from
# here via [tool.setuptools.dynamic].
__version__ = "0.2.0"

# Without a handler, logging falls back to logging.lastResort, which prints
# every WARNING-and-above record (tracebacks included) straight to stderr.
# Provider failures are already reported as error panels, so the raw
# traceback is noise unless the user asked for it with `owl -v`.
logging.getLogger(__name__).addHandler(logging.NullHandler())
