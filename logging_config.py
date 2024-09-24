import logging

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def setup_logging(level="INFO"):
    """Configure logging for the entire project."""
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )
    return logging.getLogger(__name__)


# Create a global logger instance
logger = setup_logging()
