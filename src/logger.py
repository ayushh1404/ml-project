import logging
import os
from datetime import datetime

# Create the logs directory
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# Create a log file with timestamp
log_filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".log"
log_path = os.path.join(log_dir, log_filename)

# Remove all previous handlers (important in Jupyter or VSCode)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Basic configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a"),
        logging.StreamHandler()  # also prints to console
    ]
)
