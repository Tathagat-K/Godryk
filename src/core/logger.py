# app/core/logger.py
import logging
import os
from logging.handlers import RotatingFileHandler
from src.core.config import settings

# Ensure log directory exists
log_dir = settings.LOG_DIR or 'logs'
os.makedirs(log_dir, exist_ok=True)

# Log file path
log_file = os.path.join(log_dir, 'app.log')

# Formatter for all handlers
def _get_formatter():
    return logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Console handler
def _get_console_handler():
    ch = logging.StreamHandler()
    ch.setLevel(settings.LOG_LEVEL)
    ch.setFormatter(_get_formatter())
    return ch

# Rotating file handler
def _get_file_handler():
    fh = RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding='utf-8'
    )
    fh.setLevel(settings.LOG_LEVEL)
    fh.setFormatter(_get_formatter())
    return fh

# Configure root logger
def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)

    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(_get_console_handler())
        logger.addHandler(_get_file_handler())

# Initialize configuration on import
configure_logging()

# Provide module-level logger
logger = logging.getLogger(__name__)
