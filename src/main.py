from fastapi import FastAPI
from src.api.v1 import endpoints
from src.db.database import Base, engine
from src.db.Models import product_models as _product_models  # важно импортировать модели до create_all

app = FastAPI(title="IZEN API", version="0.1.0")

Base.metadata.create_all(bind=engine)
app.include_router(endpoints.router, prefix="/api/v1")
