from __future__ import annotations
from typing import List
from pydantic import BaseModel, Field, field_validator

class ItemOrder(BaseModel):
    """Позиция в заказе."""
    name: str = Field(..., description="Название товара")
    quantity: int = Field(1, ge=1, description="Количество")
    price: float = Field(..., ge=0, description="Цена за единицу (числом)")

    @field_validator("price", mode="before")
    @classmethod
    def _price_to_float(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip().replace(" ", "").replace(",", ".")
        return float(s)

class Order(BaseModel):
    """Данные для формирования заказа."""
    too_name: str = Field(..., description="Название ТОО")
    order_data: str = Field(..., description="Дата доставки")
    client_name: str = Field(..., description="Имя клиента")
    client_number: str = Field(..., description="Телефон клиента в формате +7XXXXXXXXXX")
    delivery_address: str = Field(..., description="Адрес доставки")
    payment: str = Field(..., description="Метод оплаты")
    items: List[ItemOrder] = Field(..., min_items=1, description="Список позиций")
    comment: str = Field(..., description="Комментарий Срочно/несрочно")