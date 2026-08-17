import logging
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def setup_logger(process_name="pipeline"):
    """
    Configures global logging to output to both the terminal and a timestamped file.

    Args:
        process_name (str): The prefix for the log file (e.g., 'train', 'eval').
    """
    # Create the logs directory if it doesn't exist
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate a unique filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{process_name}_{timestamp}.log"

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),  # Terminal output
            logging.FileHandler(log_file, mode='a', encoding='utf-8')  # File output
        ]
    )
