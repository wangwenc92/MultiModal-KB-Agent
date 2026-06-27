import sys
from loguru import logger
from app.config import get_settings


def setup_logger():
    settings = get_settings()
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        "logs/app.log",
        level=settings.LOG_LEVEL,
        rotation="10 MB",
        retention="7 days",
        encoding="utf-8",
    )
    return logger


log = setup_logger()
