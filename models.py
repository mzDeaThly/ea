from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Integer, DateTime, Boolean, Float, func

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    session_token: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped["DateTime"] = mapped_column(DateTime, server_default=func.now())

class TradeLog(Base):
    __tablename__ = "trade_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    account_id: Mapped[str | None] = mapped_column(String(128), index=True)
    symbol: Mapped[str] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(8))
    lot: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    sl: Mapped[float | None] = mapped_column(Float, nullable=True)
    tp: Mapped[float | None] = mapped_column(Float, nullable=True)
    profit: Mapped[float | None] = mapped_column(Float, nullable=True)
    result: Mapped[str | None] = mapped_column(String(64))  # OPEN / CLOSE
    created_at: Mapped["DateTime"] = mapped_column(DateTime, server_default=func.now())

class LineTarget(Base):
    __tablename__ = "line_targets"
    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))   # "user" หรือ "group"
    target_id: Mapped[str] = mapped_column(String(128), index=True)  # userId / groupId
    label: Mapped[str | None] = mapped_column(String(255))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped["DateTime"] = mapped_column(DateTime, server_default=func.now())
