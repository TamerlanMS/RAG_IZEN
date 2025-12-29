from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker, declarative_base
from src.settings.db_settings import settings

Base = declarative_base()

engine = create_engine(
    settings.SYNC_DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
