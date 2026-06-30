"""数据库初始化脚本 - 创建所有表"""
import sys
sys.path.insert(0, ".")

from app.models.database import engine, Base
from app.utils.logger import log


def init_database():
    log.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    log.info("Database tables created successfully.")


if __name__ == "__main__":
    init_database()
