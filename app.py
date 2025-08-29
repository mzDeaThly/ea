from fastapi import FastAPI, Request, Depends, Form, HTTPException, UploadFile, File, Body
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.hash import bcrypt
import os, secrets, datetime as dt, json, requests
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, Session
from models import Base, User, TradeLog, LineTarget
from dotenv import load_dotenv
import joblib, numpy as np

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ai_agent.db")
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Admin#12345")
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN", "")
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
BASE_URL = os.getenv("BASE_URL", "")
DAILY_SUMMARY_TOKEN = os.getenv("DAILY_SUMMARY_TOKEN", "")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

app = FastAPI(title="MT4 AI Agent Server (No License)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# -------------------- ML Model --------------------
MODEL = None
MODEL_FEATURES = ["EMAfast","EMAslow","RSI","MACD","ATR","ADX","StochK","StochD","spread","hour"]
MODEL_CLASSES = []
MODEL_MTIME = None

def load_model():
    global MODEL, MODEL_CLASSES, MODEL_MTIME
    try:
        if not os.path.exists(MODEL_PATH):
            MODEL = None; MODEL_CLASSES = []; MODEL_MTIME = None
            print("Model file not found:", MODEL_PATH)
            return False
        MODEL = joblib.load(MODEL_PATH)
        MODEL_CLASSES = list(getattr(MODEL, "classes_", []))
        MODEL_MTIME = os.path.getmtime(MODEL_PATH)
        print("Loaded ML model:", MODEL_PATH, "classes:", MODEL_CLASSES)
        return True
    except Exception as e:
        print("ML model load failed:", e)
        MODEL = None; MODEL_CLASSES = []; MODEL_MTIME = None
        return False

# -------------------- DB INIT --------------------
def init_db():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        admin = db.scalar(select(User).where(User.email == ADMIN_EMAIL))
        if not admin:
            admin = User(email=ADMIN_EMAIL, password_hash=bcrypt.hash(ADMIN_PASSWORD), is_admin=True)
            db.add(admin)
        db.commit()

@app.on_event("startup")
def on_startup():
    init_db()
    load_model()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# -------------------- Session --------------------
SESSION_COOKIE = "aiagent_session"
def get_session(request: Request): return request.cookies.get(SESSION_COOKIE)
def require_admin(request: Request, db: Session = Depends(get_db)):
    token = get_session(request)
    if not token: raise HTTPException(status_code=302, detail="Login required")
    user = db.scalar(select(User).where(User.session_token == token, User.is_admin == True))
    if not user: raise HTTPException(status_code=302, detail="Login required")
    return user

# -------------------- LINE Push (Text + Flex) --------------------
def line_push_raw(payload: dict):
    if not LINE_CHANNEL_ACCESS_TOKEN: return
    try:
        url = "https://api.line.me/v2/bot/message/push"
        headers = {"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}","Content-Type":"application/json"}
        requests.post(url, headers=headers, json=payload, timeout=8)
    except Exception as e: print("LINE push failed:", e)

def line_push_text(to: str, text: str):
    line_push_raw({"to": to, "messages":[{"type":"text","text":text}]})

def line_flex_open_trade_contents(symbol: str, action: str, lot: float, price: float, sl, tp):
    color = "#22c55e" if action.upper()=="BUY" else "#ef4444" if action.upper()=="SELL" else "#9ca3af"
    safe_sl = "-" if sl is None else f"{sl:.5f}" if isinstance(sl, float) else str(sl)
    safe_tp = "-" if tp is None else f"{tp:.5f}" if isinstance(tp, float) else str(tp)
    now_iso = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    footer_buttons = []
    if BASE_URL:
        footer_buttons.append({
            "type":"button","style":"primary",
            "action":{"type":"uri","label":"📊 View Dashboard","uri":f"{BASE_URL}/admin/logs"}
        })
    contents = {
      "type": "bubble","size":"mega",
      "header": {"type":"box","layout":"vertical","contents":[
          {"type":"text","text":"NEW ORDER","weight":"bold","color":"#111111","size":"sm","letterSpacing":"1px"},
          {"type":"text","text": f"{symbol}", "weight":"bold","size":"xl"},
      ]},
      "body": {"type":"box","layout":"vertical","spacing":"lg","contents":[
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Action","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": action.upper(),"weight":"bold","size":"md","color": color,"flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Lot","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": f"{lot:.2f}","size":"md","flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Price","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": f"{price:.5f}","size":"md","flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"SL / TP","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": f"{safe_sl}  /  {safe_tp}","size":"md","flex":4}]},
          {"type":"separator"},
          {"type":"text","text": now_iso, "size":"xs", "color":"#9ca3af", "align":"end"}
      ]},
      "footer":{"type":"box","layout":"vertical","spacing":"sm","contents": footer_buttons or [{"type":"spacer","size":"sm"}]},
      "styles": {"footer":{"separator":True}}
    }
    return contents

def line_push_flex_open_trade(to: str, symbol: str, action: str, lot: float, price: float, sl, tp):
    contents = line_flex_open_trade_contents(symbol, action, lot, price, sl, tp)
    payload = {"to": to,"messages":[{"type":"flex","altText": f"OPEN {symbol} {action} lot={lot:.2f}","contents": contents}]}
    line_push_raw(payload)

def broadcast_open_trade(db: Session, text: str, symbol: str, action: str, lot: float, price: float, sl, tp):
    targets = db.execute(select(LineTarget).where(LineTarget.enabled==True)).scalars().all()
    seen = set()
    for t in targets:
        if t.target_id in seen: continue
        seen.add(t.target_id)
        try: line_push_flex_open_trade(t.target_id, symbol, action, lot, price, sl, tp)
        except Exception as e:
            print("Flex OPEN push failed; fallback text:", e)
            line_push_text(t.target_id, text)

# ---- Flex for CLOSE trade ----
def line_flex_close_trade_contents(symbol: str, action: str, lot: float, profit: float, open_price: float | None):
    color = "#22c55e" if profit is not None and profit >= 0 else "#ef4444"
    now_iso = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    open_txt = "-" if open_price is None else f"{open_price:.5f}"
    contents = {
      "type": "bubble", "size":"mega",
      "header": {"type":"box","layout":"vertical","contents":[
          {"type":"text","text":"ORDER CLOSED","weight":"bold","color":"#111111","size":"sm","letterSpacing":"1px"},
          {"type":"text","text": f"{symbol}", "weight":"bold","size":"xl"},
      ]},
      "body": {"type":"box","layout":"vertical","spacing":"lg","contents":[
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Action","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": action.upper(),"weight":"bold","size":"md","flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Lot","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": f"{lot:.2f}","size":"md","flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Open Price","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": open_txt,"size":"md","flex":4}]},
          {"type":"box","layout":"baseline","contents":[
              {"type":"text","text":"Profit","size":"sm","color":"#6b7280","flex":2},
              {"type":"text","text": (f"+{profit:.2f}" if (profit is not None and profit>=0) else f"{profit:.2f}" if profit is not None else "-"),"weight":"bold","size":"md","color": color,"flex":4}]},
          {"type":"separator"},
          {"type":"text","text": now_iso, "size":"xs", "color":"#9ca3af", "align":"end"}
      ]},
      "footer":{"type":"box","layout":"vertical","spacing":"sm","contents":[
          {"type":"button","style":"primary","action":{"type":"uri","label":"📊 View Dashboard","uri": f"{BASE_URL}/admin/logs" if BASE_URL else "https://line.me"}}
      ]},
      "styles": {"footer":{"separator":True}}
    }
    return contents

def line_push_flex_close_trade(to: str, symbol: str, action: str, lot: float, profit: float, open_price: float | None):
    contents = line_flex_close_trade_contents(symbol, action, lot, profit, open_price)
    payload = {"to": to,"messages":[{"type":"flex","altText": f"CLOSE {symbol} {action} P/L={profit:.2f}" if profit is not None else f"CLOSE {symbol} {action}",
                                     "contents": contents}]}
    line_push_raw(payload)

def broadcast_close_trade(db: Session, text: str, symbol: str, action: str, lot: float, profit: float, open_price: float | None):
    targets = db.execute(select(LineTarget).where(LineTarget.enabled==True)).scalars().all()
    seen = set()
    for t in targets:
        if t.target_id in seen: continue
        seen.add(t.target_id)
        try: line_push_flex_close_trade(t.target_id, symbol, action, lot, profit, open_price)
        except Exception as e:
            print("Flex CLOSE push failed; fallback text:", e)
            line_push_text(t.target_id, text)

# ---- Daily summary Flex ----
def build_daily_summary(db: Session, date_utc: dt.date | None = None):
    if date_utc is None:
        date_utc = dt.datetime.utcnow().date()
    start = dt.datetime.combine(date_utc, dt.time.min)
    end = dt.datetime.combine(date_utc, dt.time.max)
    total_trades = db.scalar(select(func.count()).select_from(TradeLog).where(TradeLog.created_at>=start, TradeLog.created_at<=end)) or 0
    closed = db.execute(select(TradeLog).where(TradeLog.created_at>=start, TradeLog.created_at<=end, TradeLog.result=="CLOSE")).scalars().all()
    total_closed = len(closed)
    pnl_sum = float(sum([c.profit or 0.0 for c in closed]))
    wins = sum(1 for c in closed if (c.profit or 0.0) > 0)
    losses = sum(1 for c in closed if (c.profit or 0.0) < 0)
    from collections import defaultdict
    sym_pnl = defaultdict(float)
    for c in closed:
        sym_pnl[c.symbol] += float(c.profit or 0.0)
    top = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]

    def fmt_p(x): return f"+{x:.2f}" if x>=0 else f"{x:.2f}"
    lines = [
        {"type":"text","text": f"Date (UTC): {date_utc.isoformat()}", "size":"sm", "color":"#6b7280"},
        {"type":"text","text": f"Closed Trades: {total_closed} / Total Logs: {total_trades}", "size":"sm"},
        {"type":"text","text": f"PnL: {fmt_p(pnl_sum)} | W: {wins} / L: {losses}", "size":"sm","weight":"bold"},
        {"type":"separator"},
        {"type":"text","text":"Top Symbols:", "size":"sm","weight":"bold"},
    ]
    if top:
        for sym, pv in top:
            lines.append({"type":"box","layout":"baseline","contents":[
                {"type":"text","text": sym, "size":"sm","flex":3},
                {"type":"text","text": fmt_p(pv), "size":"sm","flex":2, "align":"end", "color": ("#22c55e" if pv>=0 else "#ef4444")}
            ]})
    else:
        lines.append({"type":"text","text":"(no closed trades)", "size":"sm", "color":"#9ca3af"})
    bubble = {
      "type":"bubble","size":"mega",
      "header":{"type":"box","layout":"vertical","contents":[
         {"type":"text","text":"DAILY SUMMARY","size":"sm","weight":"bold","letterSpacing":"1px"},
         {"type":"text","text":"MT4 AI Agent","size":"xl","weight":"bold"}
      ]},
      "body":{"type":"box","layout":"vertical","spacing":"sm","contents": lines},
      "footer":{"type":"box","layout":"vertical","spacing":"sm","contents":[
        {"type":"button","style":"primary","action":{"type":"uri","label":"📊 View Dashboard","uri": f"{BASE_URL}" if BASE_URL else "https://line.me"}}
      ]},
      "styles":{"footer":{"separator":True}}
    }
    return bubble

def push_daily_summary(db: Session, date_utc: dt.date | None = None):
    bubble = build_daily_summary(db, date_utc)
    payload_template = {"messages":[{"type":"flex","altText":"Daily Summary","contents": bubble}]}
    targets = db.execute(select(LineTarget).where(LineTarget.enabled==True)).scalars().all()
    seen = set()
    for t in targets:
        if t.target_id in seen: continue
        seen.add(t.target_id)
        body = dict(payload_template)
        body["to"] = t.target_id
        line_push_raw(body)

# -------------------- Pages --------------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    total_users = db.scalar(select(func.count()).select_from(User)) or 0
    total_logs = db.scalar(select(func.count()).select_from(TradeLog)) or 0

    recent_logs = db.execute(select(TradeLog).order_by(TradeLog.created_at.desc()).limit(15)).scalars().all()

    # Top 10 Symbols by Profit
    topq = db.execute(
        select(TradeLog.symbol, func.coalesce(func.sum(TradeLog.profit), 0.0))
        .where(TradeLog.result == "CLOSE")
        .group_by(TradeLog.symbol)
        .order_by(func.coalesce(func.sum(TradeLog.profit), 0.0).desc())
        .limit(10)
    ).all()
    top_labels = [row[0] or "-" for row in topq]
    top_values = [float(row[1] or 0) for row in topq]

    # Daily P&L (UTC)
    pnlq = db.execute(
        select(func.date(TradeLog.created_at), func.coalesce(func.sum(TradeLog.profit), 0.0))
        .where(TradeLog.result == "CLOSE")
        .group_by(func.date(TradeLog.created_at))
        .order_by(func.date(TradeLog.created_at))
    ).all()
    pnl_labels = [str(row[0]) for row in pnlq]
    pnl_values = [float(row[1] or 0) for row in pnlq]

    # Trade count per day
    cntq = db.execute(
        select(func.date(TradeLog.created_at), func.count())
        .group_by(func.date(TradeLog.created_at))
        .order_by(func.date(TradeLog.created_at))
    ).all()
    cnt_labels = [str(row[0]) for row in cntq]
    cnt_values = [int(row[1] or 0) for row in cntq]

    # model status
    model_loaded = MODEL is not None
    model_mtime = dt.datetime.fromtimestamp(MODEL_MTIME).isoformat() if MODEL_MTIME else None

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "total_users": total_users,
        "total_logs": total_logs,
        "recent_logs": recent_logs,
        "top_labels": json.dumps(top_labels),
        "top_values": json.dumps(top_values),
        "pnl_labels": json.dumps(pnl_labels),
        "pnl_values": json.dumps(pnl_values),
        "cnt_labels": json.dumps(cnt_labels),
        "cnt_values": json.dumps(cnt_values),
        "model_loaded": model_loaded,
        "model_path": MODEL_PATH,
        "model_mtime": model_mtime,
    })

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def do_login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == email))
    if not user or not bcrypt.verify(password, user.password_hash):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"}, status_code=401)
    token = secrets.token_hex(24)
    user.session_token = token
    db.add(user); db.commit()
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(SESSION_COOKIE, token, httponly=True, max_age=7*24*3600)
    return resp

@app.get("/logout")
def logout(request: Request, db: Session = Depends(get_db)):
    token = get_session(request)
    if token:
        user = db.scalar(select(User).where(User.session_token == token))
        if user:
            user.session_token = None
            db.add(user); db.commit()
    resp = RedirectResponse(url="/login", status_code=302)
    resp.delete_cookie(SESSION_COOKIE)
    return resp

@app.get("/admin/logs", response_class=HTMLResponse)
def logs_page(request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    logs = db.execute(select(TradeLog).order_by(TradeLog.created_at.desc()).limit(300)).scalars().all()
    return templates.TemplateResponse("logs.html", {"request": request, "logs": logs})

@app.get("/admin/line", response_class=HTMLResponse)
def line_targets_page(request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    targets = db.execute(select(LineTarget).order_by(LineTarget.created_at.desc())).scalars().all()
    return templates.TemplateResponse("line.html", {"request": request, "targets": targets, "token_set": bool(LINE_CHANNEL_ACCESS_TOKEN)})

@app.post("/admin/line/new")
def line_targets_new(request: Request, kind: str = Form(...), target_id: str = Form(...), label: str = Form(""),
                     db: Session = Depends(get_db), admin=Depends(require_admin)):
    t = LineTarget(kind=kind, target_id=target_id, label=label or None, enabled=True)
    db.add(t); db.commit()
    return RedirectResponse(url="/admin/line", status_code=302)

# -------- Model Admin ---------
@app.get("/admin/model", response_class=HTMLResponse)
def model_page(request: Request, admin=Depends(require_admin)):
    model_loaded = MODEL is not None
    model_mtime = dt.datetime.fromtimestamp(MODEL_MTIME).isoformat() if MODEL_MTIME else None
    return templates.TemplateResponse("model.html", {"request": request, "model_loaded": model_loaded, "model_path": MODEL_PATH, "model_mtime": model_mtime, "model_classes": getattr(MODEL, "classes_", [])})

@app.post("/admin/model/reload")
def model_reload(admin=Depends(require_admin)):
    load_model()
    return RedirectResponse(url="/admin/model", status_code=302)

@app.post("/admin/model/upload")
async def model_upload(file: UploadFile = File(...), admin=Depends(require_admin)):
    fname = file.filename or "model.pkl"
    if not fname.endswith(".pkl"):
        raise HTTPException(status_code=400, detail="Please upload a .pkl file")
    content = await file.read()
    with open(MODEL_PATH, "wb") as f: f.write(content)
    load_model()
    return RedirectResponse(url="/admin/model", status_code=302)

# -------- Webhook helper (optional; prints userId/groupId) --------
@app.post("/webhook")
def line_webhook(payload: dict = Body(...)):
    try:
        events = payload.get("events", [])
        for ev in events:
            src = ev.get("source", {})
            uid = src.get("userId"); gid = src.get("groupId")
            print("[LINE webhook] userId:", uid, "groupId:", gid)
    except Exception as e:
        print("webhook parse error:", e)
    return {"ok": True}

# -------------------- API Schemas (no license) --------------------
class PredictRequest(BaseModel):
    symbol: str
    timeframe: str
    features: dict
    price: float
    account_id: str | None = None

class PredictResponse(BaseModel):
    action: str
    sl_pips: int
    tp_pips: int
    confidence: float

class TradeLogRequest(BaseModel):
    account_id: str | None = None
    symbol: str
    action: str
    lot: float
    price: float
    sl: float | None = None
    tp: float | None = None
    profit: float | None = None
    result: str | None = None  # OPEN/CLOSE

# -------------------- Policies --------------------
def simple_policy(features: dict) -> tuple[str, int, int, float]:
    try:
        ema_fast = float(features.get("EMAfast", 0))
        ema_slow = float(features.get("EMAslow", 0))
        rsi = float(features.get("RSI", 50))
        macd = float(features.get("MACD", 0))
        atr = max(0.00001, float(features.get("ATR", 0.0002)))
        adx = float(features.get("ADX", 20))
        stoch_k = float(features.get("StochK", 50))
        stoch_d = float(features.get("StochD", 50))
    except Exception:
        return "HOLD", 50, 100, 0.1

    trend = 1 if ema_fast > ema_slow else -1 if ema_fast < ema_slow else 0
    momentum = 1 if macd > 0 else -1 if macd < 0 else 0
    rsi_sig = 1 if rsi > 55 else -1 if rsi < 45 else 0
    stoch_cross = 1 if stoch_k > stoch_d and stoch_k < 80 else -1 if stoch_k < stoch_d and stoch_k > 20 else 0

    vote = trend + momentum + rsi_sig + stoch_cross
    if adx < 18: vote = 0 if abs(vote) <= 2 else vote//2

    if vote >= 2: action = "BUY"
    elif vote <= -2: action = "SELL"
    else: action = "HOLD"

    sl_pips = int(atr * 100000 * (2.0 if adx >= 25 else 1.2))
    tp_pips = int(atr * 100000 * (3.2 if adx >= 25 else 2.0))
    sl_pips = max(sl_pips, 15); tp_pips = max(tp_pips, sl_pips + 10)
    confidence = min(0.95, max(0.05, abs(vote) / 4.0))
    return action, sl_pips, tp_pips, confidence

def model_policy(features: dict) -> tuple[str, int, int, float]:
    if MODEL is None: return simple_policy(features)
    try:
        x = np.array([[float(features.get(k, 0)) for k in MODEL_FEATURES]])
        pred = MODEL.predict(x)[0]
        proba = getattr(MODEL, "predict_proba", lambda X: [[0.33,0.33,0.34]])(x)[0]
        classes = list(getattr(MODEL, "classes_", ["BUY","SELL","HOLD"]))
        prob_map = {cls: prob for cls, prob in zip(classes, proba)}
        conf = float(prob_map.get(pred, max(proba)))
        atr = max(0.00001, float(features.get("ATR", 0.0002)))
        adx = float(features.get("ADX", 20))
        sl_pips = int(atr * 100000 * (2.4 if adx >= 25 else 1.4))
        tp_pips = int(atr * 100000 * (3.6 if adx >= 25 else 2.2))
        sl_pips = max(sl_pips, 15); tp_pips = max(tp_pips, sl_pips + 10)
        return pred, sl_pips, tp_pips, conf
    except Exception as e:
        print("model_policy error:", e); return simple_policy(features)

# -------------------- API (no license) --------------------
@app.post("/api/v1/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    action, sl_pips, tp_pips, conf = model_policy(req.features or {})
    return PredictResponse(action=action, sl_pips=sl_pips, tp_pips=tp_pips, confidence=conf)

@app.post("/api/v1/trade_log")
def api_trade_log(req: TradeLogRequest, db: Session = Depends(get_db)):
    row = TradeLog(
        account_id=req.account_id,
        symbol=req.symbol,
        action=req.action,
        lot=req.lot,
        price=req.price,
        sl=req.sl,
        tp=req.tp,
        result=req.result,
        profit=req.profit
    )
    db.add(row); db.commit()

    if req.result and req.result.upper() == "OPEN":
        msg = f"📣 OPEN {req.symbol} {req.action} lot={req.lot:.2f} @ {req.price:.5f} | SL={req.sl or '-'} TP={req.tp or '-'}"
        broadcast_open_trade(db, msg, req.symbol, req.action, req.lot, req.price, req.sl, req.tp)
    if req.result and req.result.upper() == "CLOSE":
        msg2 = f"✅ CLOSE {req.symbol} {req.action} lot={req.lot:.2f} P/L={req.profit:.2f}"
        broadcast_close_trade(db, msg2, req.symbol, req.action, req.lot, req.profit or 0.0, req.price)
    return {"ok": True}

# ---- Protected endpoint to push daily summary ----
@app.post("/api/v1/push_daily_summary")
def api_push_daily_summary(token: str, date: str | None = None, db: Session = Depends(get_db)):
    if not DAILY_SUMMARY_TOKEN or token != DAILY_SUMMARY_TOKEN:
        raise HTTPException(status_code=403, detail="Forbidden")
    d = None
    if date:
        try:
            d = dt.datetime.strptime(date, "%Y-%m-%d").date()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date, use YYYY-MM-DD")
    push_daily_summary(db, d)
    return {"ok": True, "date": (d.isoformat() if d else dt.datetime.utcnow().date().isoformat())}
