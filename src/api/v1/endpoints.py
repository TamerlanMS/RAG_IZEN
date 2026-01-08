from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session
from starlette import status
from starlette.requests import Request

from src.common.logger import logger
from src.common.tools.ReAct_agent import agent
from src.db.CRUD import create_db, update_db
from src.db.database import get_db

router: APIRouter = APIRouter()
logger.info("Starting app .....")

# ---------------- Agent ---------------- #

@router.get("/ask_llm", tags=["Agent"])
async def ask_agent(
    request: Request,
) -> Any:
    try:
        body = await request.json()
        user_input = body.get("user_input", None)
        thread_id = body.get("thread_id", None)
        if user_input and thread_id:
            inputs = {"messages": [("user", user_input)]}
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100
            }
        else:
            raise ValueError

    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON body"
        )
    except Exception as e:
        logger.error("Unexpected error with request body - %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )

    try:
        answer = agent.invoke(inputs, config=config)
        ai_answer = answer["messages"][-1].content
    except AttributeError:
        logger.warning("Unexpected response format from agent", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected response format from agent",
        )
    except Exception as e:
        logger.error("Unexpected error with answer from LLM - %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )

    return {"answer": ai_answer}


# ---------------- DB utils ---------------- #

@router.get("/status_DB", tags=["database"])
async def get_postgres_db_status(
    request: Request,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Возвращает версию Postgres. Полезно для health-check.
    """
    try:
        version = db.scalar(text("SELECT version();"))
        return {"status": status.HTTP_200_OK, "DB_version": version}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error connecting to DB: {e}",
        )


@router.post("/create_DB", tags=["database"])
async def create_tables() -> Dict[str, Any]:
    """
    Создаёт таблицы (products) при необходимости.
    """
    try:
        message = create_db()
        return {"status_code": status.HTTP_200_OK, "transaction": message}
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Error creating tables"
        )


@router.post("/update_DB", tags=["database"])
async def update_products(
    payload: Optional[dict] = Body(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Обновление БД продуктами.
    Пустой POST — подтянет JSON по умолчанию: https://ts23.cloud1c.pro/FileGPT/GMZProducts.json
    С телом — ожидает объект:
    {
      "Date": "2025-09-09T10:14:58Z",
      "Products": [
        { "id": "00-114.1", "name": "…", "price": "1" },
        { "id": "00-155",   "name": "…", "price": "240" }
      ]
    }
    """
    try:
        if payload is not None and "Products" not in payload:
            raise ValueError("Expected JSON object with key 'Products'")
        total = update_db(db, json_data=payload) if payload is not None else update_db(db)
        return {
            "status_code": status.HTTP_202_ACCEPTED,
            "message": f"Total updated: {total}",
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {e}",
        )
