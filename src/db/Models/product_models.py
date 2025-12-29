from sqlalchemy import Column, Integer, String
from src.db.database import Base

class Product(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    external_id = Column(String, unique=True, index=True, nullable=True)  # из JSON "id"
    name = Column(String, unique=True, index=True, nullable=False)
    price = Column(String, nullable=False)  # храним как пришло (строкой)
