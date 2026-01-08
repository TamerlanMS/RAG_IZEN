from __future__ import annotations

import re
from itertools import count
from typing import Annotated, Any, List, Optional, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode
from datetime import datetime, timezone as dt_timezone, timedelta
from typing import Optional

from src.common.llm_model import LLM
from src.common.Schemas.product_schemas import ItemOrder, Order
from src.common.vector_store import vector_store
from src.db.CRUD import get_product_price, get_products_by_name, get_product_price_by_name
from src.db.database import get_db
from src.settings.config import AGENT_PROMPT

load_dotenv()

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:
    ZoneInfo = None

@tool
def get_current_time(
    timezone: Optional[str] = "Asia/Qyzylorda",
    format: str = "iso",
    include_components: bool = True,
) -> dict:
    """
    Вернуть текущее время/дату.

    Аргументы:
      - timezone: IANA-таймзона (например, 'Asia/Qyzylorda'). Неверная -> UTC.
      - format: 'iso' | 'rfc3339' | 'unix' | 'custom'
      - include_components: добавить ли разложение по частям (date, time, year...).

    Возвращает JSON-совместимый словарь:
      {
        "result": <строка/число>,
        "timezone": <строка>,
        "iso_utc": <ISO в UTC c Z>,
        "unix": <int unix seconds>,
        "components": {...}  # если include_components=True
      }
    """

    # --- таймзона ---
    tz = dt_timezone.utc
    tz_name = "UTC"
    if timezone and ZoneInfo is not None:
        try:
            tz = ZoneInfo(timezone)  # type: ignore[arg-type]
            tz_name = timezone
        except Exception:
            tz = dt_timezone.utc
            tz_name = "UTC"

    now = datetime.now(tz)
    now_utc = now.astimezone(dt_timezone.utc)

    # --- основное поле result ---
    fmt = (format or "iso").lower()
    if fmt == "iso":
        result = now.isoformat()
    elif fmt == "rfc3339":
        s = now.isoformat()
        # если UTC — делаем Z
        if now.utcoffset() == timedelta(0):
            s = now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z")
        result = s
    elif fmt == "unix":
        result = int(now.timestamp())
    elif fmt == "custom":
        # Можно заменить под нужный формат, если надо
        result = now.strftime("%Y-%m-%d %H:%M:%S")
    else:
        result = now.isoformat()

    out = {
        "result": result,
        "timezone": tz_name,
        "iso_utc": now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "unix": int(now.timestamp()),
    }

    if include_components:
        out["components"] = {
            "date": now.date().isoformat(),
            "time": now.time().replace(microsecond=0).isoformat(),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            # +HH:MM
            "utc_offset": (
                ("+" if (now.utcoffset() or timedelta(0)) >= timedelta(0) else "-")
                + f"{abs(int((now.utcoffset() or timedelta(0)).total_seconds())) // 3600:02d}:"
                + f"{(abs(int((now.utcoffset() or timedelta(0)).total_seconds())) % 3600) // 60:02d}"
            ),
        }

    return out

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool  # type: ignore
def add(a: int, b: int) -> int:
    """Сложить два целых числа."""
    return a + b

@tool  # type: ignore
def check_phone_number(phone_number: str) -> Optional[str]:
    """
    Приводит номер к формату +7XXXXXXXXXX.
    Возвращает нормализованный номер или None.
    """
    cleaned = re.sub(r"[^\d+]", "", phone_number)
    if cleaned.startswith("+7"):
        number = cleaned[2:]
    elif cleaned.startswith("8"):
        number = cleaned[1:]
    else:
        return None
    if len(number) == 10 and number.isdigit():
        return f"+7{number}"
    return None

@tool  # type: ignore
def find_product_in_vector_store(product_name: str) -> Any:
    """
    Найти похожие товары: сначала БД по подстроке, если пусто — векторный поиск.
    """
    db_search_result = get_products_by_name(product_name)
    if not db_search_result:
        return vector_store.search(product_name)
    return db_search_result

@tool
def get_current_price(product_name: str):
    """
    Верни текущую цену товара по названию (поиск нечувствителен к регистру и неполному совпадению).
    Возвращает: {"name": "...", "external_id": "...", "price": ...} или None.
    """
    db = next(get_db())
    return get_product_price_by_name(db, product_name)

@tool(parse_docstring=True, args_schema=Order)  # type: ignore
def create_order(
    too_name: str,
    order_data: str,
    client_name: str,
    client_number: str,
    delivery_address: str,
    payment: str,
    items: List[ItemOrder],
    comment: str
) -> str:
    """
    Сформировать текст заказа.
    Требуются: Название ТОО, ФИО, Телефон, Адрес доставки, Дата доставки, Список позиций.
    """
    total = 0
    lines: List[str] = []
    counter = 1
    total_products = 0
    for it in items:
        price = int(round(float(it.price)))
        qty = int(it.quantity)
        line_sum = price * qty
        total_products += line_sum
        lines.append(
            f"№{counter}: {it.name}\n"
            f"Цена: {price} тг/шт.\n"
            f"Количество: {qty} шт.\n"
            f"Сумма: {line_sum} тг"
        )
        counter += 1
        
    lines_str = "\n".join(lines)
    date_str = order_data
    comment = comment or "несрочно"
    delivery = ""
    if total>50000:
        delivery = "бесплатно"
    else:
        delivery = "платная"


    return (
        "Ваш заказ:\n"
        f"Название ТОО: {too_name}\n"
        f"ФИО: {client_name}\n"
        f"Телефон: {client_number}\n"
        f"Адрес доставки: {delivery_address}\n"
        f"Метод оплаты: {payment}\n"
        f"Дата доставки: {date_str}\n"
        f"Товары:\n{lines_str}\n\n"
        f"Итого к оплате: {total:.2f}\n"
        f"Доставка: {delivery}\n"
        f"Комментарий: {comment}\n"
        f"Код подтверждения: ti5Q9L2PJ"
        "❗Для подтверждения заказа обязательно нажмите на кнопку \"ОТПРАВИТЬ ЗАКАЗ\" после данного сообщения👇❗",
    )

tools: List[BaseTool] = [
    add,
    find_product_in_vector_store,
    get_current_price,
    check_phone_number,
    create_order,
    get_current_time,
]
tool_node = ToolNode(tools)
llm = LLM.bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=AGENT_PROMPT)
    response = llm.invoke([system_prompt] + list(state["messages"]))
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "continue"
    return "end"

graph = StateGraph(AgentState)
graph.add_node("agent", model_call)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
graph.add_edge("tools", "agent")

agent = graph.compile(checkpointer=InMemorySaver(), debug=True)
