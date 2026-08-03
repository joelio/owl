"""Parliament of Owls - Query multiple LLMs in parallel."""

import logging

__version__ = "0.1.0"

# Without a handler, logging falls back to logging.lastResort, which prints
# every WARNING-and-above record (tracebacks included) straight to stderr.
# Provider failures are already reported as error panels, so the raw
# traceback is noise unless the user asked for it with `owl -v`.
logging.getLogger(__name__).addHandler(logging.NullHandler())
