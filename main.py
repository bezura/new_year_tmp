import datetime
from typing import List, Optional

from sqlalchemy import JSON

from fastapi import HTTPException, Depends, Request, status
from sqlalchemy.orm import declarative_base, Mapped, mapped_column
from sqlalchemy import DateTime, func
from sqlalchemy import String, BigInteger, Integer, Date, Time, ForeignKey, Enum, Float
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import func
from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy.ext.asyncio import AsyncAttrs, async_sessionmaker, create_async_engine, AsyncSession
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute, APIRouter
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.future import select
from sqlalchemy import update as sqlalchemy_update, delete as sqlalchemy_delete
from datetime import datetime
from fastapi import APIRouter, Depends, Query

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict
from init_data_py import InitData


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(json_encoders={datetime: lambda o: o.isoformat()})


database_url = 'sqlite+aiosqlite:///db.sqlite3'
engine = create_async_engine(url=database_url)
async_session_maker = async_sessionmaker(engine, class_=AsyncSession)


class Base(AsyncAttrs, DeclarativeBase):
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class UserDB(Base):
    __tablename__ = 'users'

    telegram_id: Mapped[int] = mapped_column(BigInteger,
                                             primary_key=True)  # Уникальный идентификатор пользователя в Telegram
    first_name: Mapped[str] = mapped_column(String, nullable=False)  # Имя пользователя
    username: Mapped[str | None] = mapped_column(String, nullable=True)  # Telegram username
    score: Mapped[float | None] = mapped_column(Float, nullable=True)


class UserResponse(BaseModel):
    telegram_id: int
    first_name: str
    username: str | None
    score: float | None


class TelegramAuth(BaseModel):
    id: Optional[int] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    username: Optional[str] = None
    photo_url: Optional[str] = None
    auth_date: Optional[str] = None
    hash: Optional[str] = None


async def auth_user_request(request: Request):
    user_data_string = request.headers.get("Authorization")
    if not user_data_string:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    init_data = InitData.parse(user_data_string)
    user = init_data.user
    async with async_session_maker() as session:
        query = select(func.count(UserDB.telegram_id)).where(UserDB.telegram_id == user.id)
        result = await session.execute(query)
        ans = result.scalar()
        if ans <= 0:
            new_user = UserDB(
                telegram_id=user.id,
                first_name=user.first_name,
                username=user.username
            )
            session.add(new_user)
            try:
                await session.commit()
            except SQLAlchemyError as e:
                await session.rollback()
                raise e
    return True


app = FastAPI(title="BOT",
              openapi_url="/api/v1/openapi.json",
              debug=True,
              docs_url='/api/docs',
              redoc_url='/api/redoc',
              version='0.0.1')


@app.get("/up")
async def health_check_endpoint():
    """
    Возвращает информацию о состоянии сервера
    """
    return dict(
        status="Server is running correctly",
        server_time=datetime.datetime.now(),
        version="1.0.0"
    )


@app.get("/users", response_model=List[UserResponse])
async def get_users_endpoint(
        _=Depends(auth_user_request),
):
    async with async_session_maker() as session:
        query = select(UserDB)
        result = await session.execute(query)
        return result.scalars().all()


@app.get("/user/{telegram_id}", response_model=UserResponse)
async def get_user_by_id_endpoint(
        telegram_id: int,
        _=Depends(auth_user_request),
):
    async with async_session_maker() as session:
        query = select(UserDB).where(UserDB.telegram_id == telegram_id)
        result = await session.execute(query)
        return result.scalar()


@app.get("/user/{telegram_id}/update-score")
async def update_user_score_by_id_endpoint(
        telegram_id: int,
        _=Depends(auth_user_request),
        score: float = Query(0)
):
    async with async_session_maker() as session:
        query = sqlalchemy_update(UserDB).where(UserDB.telegram_id == telegram_id).values(score=score)
        result = await session.execute(query)
        try:
            await session.commit()
        except SQLAlchemyError as e:
            await session.rollback()
            raise e
        if not result.rowcount:
            return False
        return True


app.add_middleware(
    CORSMiddleware,
    allow_origins=["localhost", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "main:app",
#         host="0.0.0.0",
#         port=settings.PORT,
#         reload=settings.DEBUG,
#     )
