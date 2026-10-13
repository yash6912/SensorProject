
## Logging configuration

import logging
import os
from datetime import datetime

# Create the logs directory
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Generate a unique log filename
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create the complete log file path only once
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(levelname)s - %(message)s",
    level=logging.INFO,
)

logging.info("Logger initialized successfully")
