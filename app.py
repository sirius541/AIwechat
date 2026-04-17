#!/usr/bin/env python3
"""
AI WeChat - 多模型AI群聊应用
支持 GLM、Qwen、DeepSeek、Kimi、豆包 等大模型
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AI WeChat")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)
CONFIG_PATH = BASE_DIR / "config.json"
DATA_DIR = BASE_DIR / "data"
SESSIONS_PATH = DATA_DIR / "sessions.json"

# 未传 max_tokens 时，多数网关会用很小的默认值（如 1024），导致回答被截断
DEFAULT_MAX_OUTPUT_TOKENS = 8192


def effective_max_tokens(model_cfg: dict, app_config: Optional[dict] = None) -> int:
    """单模型 max_tokens > config.json 顶层 default_max_tokens > 环境变量 AIWX_MAX_TOKENS > 内置默认。"""
    app_config = app_config or {}
    for src in (model_cfg, app_config):
        key = "max_tokens" if src is model_cfg else "default_max_tokens"
        v = src.get(key)
        if v is not None:
            try:
                return max(1, int(v))
            except (TypeError, ValueError):
                pass
    try:
        return max(1, int(os.getenv("AIWX_MAX_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))))
    except ValueError:
        return DEFAULT_MAX_OUTPUT_TOKENS


DEFAULT_CONFIG = {
    "models": [
        {
            "id": "glm-4-flash",
            "name": "GLM",
            "display_name": "智谱 GLM-4",
            "emoji": "🤖",
            "color": "#07C160",
            "type": "zhipuai",
            "api_key": "",
            "description": "智谱AI，擅长中文对话与推理",
        },
        {
            "id": "qwen-plus",
            "name": "Qwen",
            "display_name": "通义千问",
            "emoji": "🧠",
            "color": "#FF6B35",
            "type": "openai_compat",
            "api_key": "",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "description": "阿里云通义千问，多模态理解能力强",
        },
        {
            "id": "deepseek-chat",
            "name": "DeepSeek",
            "display_name": "DeepSeek",
            "emoji": "🔭",
            "color": "#4A90E2",
            "type": "openai_compat",
            "api_key": "",
            "base_url": "https://api.deepseek.com",
            "description": "深度求索，强大推理与代码能力",
        },
        {
            "id": "moonshot-v1-8k",
            "name": "Kimi",
            "display_name": "Kimi",
            "emoji": "🌙",
            "color": "#7C4DFF",
            "type": "openai_compat",
            "api_key": "",
            "base_url": "https://api.moonshot.cn/v1",
            "description": "月之暗面Kimi，擅长长文本理解",
        },
        {
            "id": "doubao-lite-4k",
            "name": "Doubao",
            "display_name": "豆包",
            "emoji": "🫘",
            "color": "#FF4757",
            "type": "openai_compat",
            "api_key": "",
            "base_url": "https://ark.cn-beijing.volces.com/api/v3",
            "description": "字节跳动豆包，创意写作能力强",
        },
        {
            "id": "ERNIE-4.0-8K",
            "name": "ERNIE",
            "display_name": "文心一言",
            "emoji": "📝",
            "color": "#2979FF",
            "type": "openai_compat",
            "api_key": "",
            "base_url": "https://qianfan.baidubce.com/v2",
            "description": "百度文心一言，中文写作与知识问答",
        },
    ]
}


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    save_config(DEFAULT_CONFIG)
    return DEFAULT_CONFIG


def save_config(config: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _load_sessions_store() -> dict:
    if not SESSIONS_PATH.exists():
        return {"version": 1, "items": []}
    with open(SESSIONS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _save_sessions_store(data: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class SessionCreate(BaseModel):
    title: Optional[str] = None
    kind: str = "group"
    privateModelId: Optional[str] = None


class SessionPatch(BaseModel):
    title: Optional[str] = None


# ──────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def index():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>前端文件未找到，请检查 static/index.html</h1>")


@app.get("/api/models")
async def get_models():
    return JSONResponse(load_config()["models"])


@app.post("/api/config")
async def update_config(request: Request):
    data = await request.json()
    config = load_config()
    model_map = {m["id"]: m for m in config["models"]}
    for model_id, updates in data.items():
        if model_id in model_map:
            for k, v in updates.items():
                model_map[model_id][k] = v
    save_config(config)
    return JSONResponse({"status": "ok"})


@app.get("/api/sessions")
async def list_sessions():
    data = _load_sessions_store()
    items = []
    for s in data.get("items", []):
        items.append(
            {
                "id": s["id"],
                "title": s.get("title", "会话"),
                "kind": s.get("kind", "group"),
                "privateModelId": s.get("privateModelId"),
                "updatedAt": s.get("updatedAt", 0),
                "messageCount": len(s.get("messages", [])),
            }
        )
    items.sort(key=lambda x: -float(x["updatedAt"]))
    return JSONResponse(items)


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    data = _load_sessions_store()
    for s in data.get("items", []):
        if s["id"] == session_id:
            return JSONResponse(s)
    raise HTTPException(status_code=404, detail="session not found")


@app.post("/api/sessions")
async def create_session(body: SessionCreate):
    data = _load_sessions_store()
    now = time.time()
    sid = uuid.uuid4().hex[:16]
    title = (body.title or "").strip()
    if not title:
        if body.kind == "private" and body.privateModelId:
            title = f"私聊 · {body.privateModelId}"
        else:
            title = "新群聊"
    sess = {
        "id": sid,
        "title": title,
        "kind": body.kind,
        "privateModelId": body.privateModelId if body.kind == "private" else None,
        "messages": [],
        "createdAt": now,
        "updatedAt": now,
    }
    data.setdefault("items", []).append(sess)
    _save_sessions_store(data)
    return JSONResponse(sess)


@app.put("/api/sessions/{session_id}")
async def put_session(session_id: str, request: Request):
    payload = await request.json()
    data = _load_sessions_store()
    for s in data.get("items", []):
        if s["id"] == session_id:
            if "title" in payload:
                s["title"] = payload["title"]
            if "messages" in payload:
                s["messages"] = payload["messages"]
            s["updatedAt"] = time.time()
            _save_sessions_store(data)
            return JSONResponse(s)
    raise HTTPException(status_code=404, detail="session not found")


@app.patch("/api/sessions/{session_id}")
async def patch_session(session_id: str, body: SessionPatch):
    data = _load_sessions_store()
    for s in data.get("items", []):
        if s["id"] == session_id:
            if body.title is not None and body.title.strip():
                s["title"] = body.title.strip()
            s["updatedAt"] = time.time()
            _save_sessions_store(data)
            return JSONResponse(s)
    raise HTTPException(status_code=404, detail="session not found")


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    data = _load_sessions_store()
    items = data.get("items", [])
    new_items = [x for x in items if x["id"] != session_id]
    if len(new_items) == len(items):
        raise HTTPException(status_code=404, detail="session not found")
    data["items"] = new_items
    _save_sessions_store(data)
    return JSONResponse({"status": "ok"})


# ──────────────────────────────────────────────────────────────
# Chat models
# ──────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str = ""
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    # data:image/...;base64,... 或 https 图链，多模态对话用
    images: Optional[List[str]] = None


class ChatRequest(BaseModel):
    message: str = ""
    models: List[str]
    history: List[ChatMessage] = []
    # 当前这条用户消息的配图（与 history 分离，因 slice 不包含本轮）
    message_images: Optional[List[str]] = None
    # 群聊：把其他模型的发言一并作为上下文，便于协作与接力
    group_context: bool = False


GROUP_SYSTEM = (
    "你正在与其他多位 AI 在同一个群聊中协作。你会看到其他助手的发言，"
    "它们会以「【某某在群聊中的发言】」形式出现。你可以引用、补充、反驳或接力完成用户的问题，"
    "像真人群聊一样自然交流。"
)


def pack_user_content(text: str, images: Optional[List[str]]) -> Union[str, List[Dict[str, Any]]]:
    """OpenAI 兼容多模态：无图则纯文本，有图则 content 为 parts 数组。"""
    t = (text or "").strip() or ("请根据图片回答。" if images else "")
    urls = [
        u.strip()
        for u in (images or [])
        if isinstance(u, str)
        and (u.startswith("data:image/") or u.startswith("https://") or u.startswith("http://"))
    ][:8]
    if not urls:
        return t or " "
    parts: List[Dict[str, Any]] = [{"type": "text", "text": t or "请结合图片说明。"}]
    for u in urls:
        parts.append({"type": "image_url", "image_url": {"url": u}})
    return parts


def flatten_content_to_text(content: Any) -> str:
    """把多模态 content 压成纯文本（给不支持视觉的模型）。"""
    if isinstance(content, str):
        return content or " "
    if not isinstance(content, list):
        return str(content) if content is not None else " "
    texts: List[str] = []
    had_img = False
    for p in content:
        if not isinstance(p, dict):
            continue
        if p.get("type") == "text":
            texts.append(str(p.get("text") or ""))
        elif p.get("type") == "image_url":
            had_img = True
    body = "\n".join(texts).strip()
    if had_img:
        note = "【用户上传了图片；当前模型为纯文本，无法查看图像，仅根据文字作答】"
        return f"{note}\n{body}".strip() if body else note
    return body or " "


def strip_messages_to_text_only(messages: List[dict]) -> List[dict]:
    return [{"role": m["role"], "content": flatten_content_to_text(m.get("content"))} for m in messages]


def messages_have_images(messages: List[dict]) -> bool:
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for p in c:
                if isinstance(p, dict) and p.get("type") == "image_url":
                    return True
    return False


def _vision_heuristic(model_id: str) -> bool:
    """未配置 supports_vision 时的保守推断：仅对常见多模态命名返回 True。"""
    mid = (model_id or "").lower()
    if "deepseek" in mid and "vl" not in mid and "vision" not in mid:
        return False
    # 智谱：glm-5 / glm-4 等多为纯文本；glm-4v、带 vl 的才按多模态
    if "glm" in mid:
        return any(x in mid for x in ("4v", "glm-v", "-vl", "vl-", "vision", "multimodal"))
    # Kimi / Moonshot：kimi-k2、moonshot-v1 等默认文本；需显式 vision/vl 或配置 vision_model_id
    if "kimi" in mid or "moonshot" in mid:
        return any(x in mid for x in ("vision", "vl-", "-vl", "multimodal"))
    return True


def model_supports_vision(cfg: dict) -> bool:
    """
    是否对当前请求发送 image_url。
    可在 config 里设置：
    - supports_vision: true/false 强制开关
    - vision_model_id: 有图时改用该模型 id 调 API（主 id 仍可叫 glm-5，看图时走 4v 等）
    """
    vid = (cfg.get("vision_model_id") or "").strip()
    if cfg.get("supports_vision") is False and not vid:
        return False
    if cfg.get("supports_vision") is True:
        return True
    if vid:
        return True
    return _vision_heuristic(cfg.get("id") or "")


def resolve_api_model_id(cfg: dict, has_images: bool) -> str:
    """有图且配置了 vision_model_id 时，用视觉模型名请求网关。"""
    vid = (cfg.get("vision_model_id") or "").strip()
    if has_images and vid:
        return vid
    return cfg.get("id") or ""


def chat_completions_url(base_url: str) -> str:
    u = (base_url or "https://api.openai.com/v1").rstrip("/")
    if u.endswith("/chat/completions"):
        return u
    return f"{u}/chat/completions"


def _display_name_for(msg: ChatMessage, id_to_name: Dict[str, str]) -> str:
    if msg.model_name and msg.model_name.strip():
        return msg.model_name.strip()
    if msg.model_id and msg.model_id in id_to_name:
        return id_to_name[msg.model_id]
    return msg.model_id or "其他 AI"


def build_model_history(
    history: List[ChatMessage],
    model_id: str,
    *,
    group_context: bool,
    id_to_name: Dict[str, str],
) -> List[dict]:
    """私聊：只发用户 + 本模型回复。群聊：用户 + 本模型 assistant + 其他模型以 user 包装，便于 API 交替。"""
    result: List[dict] = []
    if group_context:
        result.append({"role": "system", "content": GROUP_SYSTEM})

    for msg in history:
        if msg.role == "user":
            result.append(
                {"role": "user", "content": pack_user_content(msg.content, msg.images)}
            )
        elif msg.role == "assistant":
            if not (msg.content or "").strip():
                continue
            if group_context:
                if msg.model_id == model_id:
                    result.append({"role": "assistant", "content": msg.content})
                else:
                    who = _display_name_for(msg, id_to_name)
                    result.append(
                        {
                            "role": "user",
                            "content": f"【{who} 在群聊中的发言】\n{msg.content}",
                        }
                    )
            elif msg.model_id == model_id:
                result.append({"role": "assistant", "content": msg.content})

    return result


# ──────────────────────────────────────────────────────────────
# Streaming helpers
# ──────────────────────────────────────────────────────────────


async def _call_zhipuai(
    model_cfg: dict,
    messages: list,
    msg_id: str,
    queue: asyncio.Queue,
    *,
    api_model_id: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
):
    try:
        api_key = model_cfg.get("api_key", "").strip()
        if not api_key:
            await queue.put(
                {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
                 "content": "API Key 未配置，请点击右上角 ⚙️ 设置"}
            )
            return

        from zhipuai import ZhipuAI  # noqa: PLC0415

        client = ZhipuAI(api_key=api_key)
        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        mid = model_cfg["id"] if api_model_id is None else api_model_id

        def sync_stream():
            try:
                for chunk in client.chat.completions.create(
                    model=mid,
                    messages=messages,
                    stream=True,
                    max_tokens=max_tokens,
                ):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", None) or ""
                    if content:
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put(("token", content)), loop
                        ).result(timeout=10)
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(("done", None)), loop
                ).result(timeout=10)
            except Exception as exc:
                asyncio.run_coroutine_threadsafe(
                    token_queue.put(("error", str(exc))), loop
                ).result(timeout=5)

        await queue.put(
            {
                "type": "start",
                "model_id": model_cfg["id"],
                "model_name": model_cfg["display_name"],
                "message_id": msg_id,
                "emoji": model_cfg["emoji"],
                "color": model_cfg["color"],
            }
        )

        t = threading.Thread(target=sync_stream, daemon=True)
        t.start()

        while True:
            kind, content = await asyncio.wait_for(token_queue.get(), timeout=120)
            if kind == "token":
                await queue.put(
                    {"type": "token", "model_id": model_cfg["id"],
                     "message_id": msg_id, "content": content}
                )
            elif kind == "done":
                break
            elif kind == "error":
                await queue.put(
                    {"type": "error", "model_id": model_cfg["id"],
                     "message_id": msg_id, "content": content}
                )
                return

        await queue.put({"type": "done", "model_id": model_cfg["id"], "message_id": msg_id})

    except ImportError:
        await queue.put(
            {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
             "content": "zhipuai 未安装，请运行: pip install zhipuai"}
        )
    except Exception as exc:
        logger.error("ZhipuAI error: %s", exc)
        await queue.put(
            {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
             "content": f"错误: {str(exc)[:120]}"}
        )


async def _call_openai_compat(
    model_cfg: dict,
    messages: list,
    msg_id: str,
    queue: asyncio.Queue,
    *,
    api_model_id: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
):
    try:
        api_key = model_cfg.get("api_key", "").strip()
        if not api_key:
            await queue.put(
                {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
                 "content": "API Key 未配置，请点击右上角 ⚙️ 设置"}
            )
            return

        post_url = chat_completions_url(model_cfg.get("base_url", "https://api.openai.com/v1"))
        mid = model_cfg["id"] if api_model_id is None else api_model_id

        await queue.put(
            {
                "type": "start",
                "model_id": model_cfg["id"],
                "model_name": model_cfg["display_name"],
                "message_id": msg_id,
                "emoji": model_cfg["emoji"],
                "color": model_cfg["color"],
            }
        )

        # 流式长输出：放宽 read 超时（两次 chunk 之间的间隔），避免长文生成被误判断连
        stream_timeout = httpx.Timeout(60.0, read=600.0, write=60.0, pool=60.0)
        payload = {
            "model": mid,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=stream_timeout) as client:
            async with client.stream(
                "POST",
                post_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    await queue.put(
                        {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
                         "content": f"HTTP {resp.status_code}: {err.decode()[:120]}"}
                    )
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:].strip()
                    if payload == "[DONE]" or not payload:
                        break
                    try:
                        chunk = json.loads(payload)
                        content = (
                            chunk.get("choices", [{}])[0]
                            .get("delta", {})
                            .get("content", "")
                        )
                        if content:
                            await queue.put(
                                {"type": "token", "model_id": model_cfg["id"],
                                 "message_id": msg_id, "content": content}
                            )
                    except json.JSONDecodeError:
                        continue

        await queue.put({"type": "done", "model_id": model_cfg["id"], "message_id": msg_id})

    except Exception as exc:
        logger.error("OpenAI compat error [%s]: %s", model_cfg["name"], exc)
        await queue.put(
            {"type": "error", "model_id": model_cfg["id"], "message_id": msg_id,
             "content": f"错误: {str(exc)[:120]}"}
        )


# ──────────────────────────────────────────────────────────────
# Chat endpoint
# ──────────────────────────────────────────────────────────────


@app.post("/api/chat")
async def chat(req: ChatRequest):
    config = load_config()
    model_map = {m["id"]: m for m in config["models"]}

    responding = [model_map[mid] for mid in req.models if mid in model_map]
    if not responding:
        raise HTTPException(status_code=400, detail="No valid models specified")

    id_to_name = {m["id"]: m.get("display_name") or m.get("name") or m["id"] for m in config["models"]}
    use_group_ctx = bool(req.group_context)

    queue: asyncio.Queue = asyncio.Queue()

    async def generate():
        tasks = []
        for model in responding:
            msg_id = uuid.uuid4().hex[:8]
            history = build_model_history(
                req.history,
                model["id"],
                group_context=use_group_ctx,
                id_to_name=id_to_name,
            )
            history.append(
                {
                    "role": "user",
                    "content": pack_user_content(req.message, req.message_images),
                }
            )
            has_images = messages_have_images(history)
            if has_images and not model_supports_vision(model):
                history = strip_messages_to_text_only(history)
            # 剥离图片后必须用主模型 id，不能仍走 vision_model_id
            api_mid = resolve_api_model_id(model, messages_have_images(history))
            max_out = effective_max_tokens(model, config)

            if model["type"] == "zhipuai":
                coro = _call_zhipuai(
                    model, history, msg_id, queue, api_model_id=api_mid, max_tokens=max_out
                )
            else:
                coro = _call_openai_compat(
                    model, history, msg_id, queue, api_model_id=api_mid, max_tokens=max_out
                )
            tasks.append(asyncio.create_task(coro))

        completed = 0
        total = len(responding)
        while completed < total:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=120)
                if event["type"] in ("done", "error"):
                    completed += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                break

        yield 'data: {"type":"all_done"}\n\n'

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    logger.info("Starting AI WeChat on http://0.0.0.0:%d", port)
    uvicorn.run(app, host="0.0.0.0", port=port)
