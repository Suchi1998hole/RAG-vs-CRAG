from loguru import logger
import os

os.makedirs("logs", exist_ok=True)
logger.add("logs/run.log", rotation="1 MB", enqueue=True)

__all__ = ["logger"]
