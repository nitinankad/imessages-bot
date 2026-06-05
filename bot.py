import os
import hmac
import hashlib
import json
import logging
import random
import time
import threading
import uuid
import urllib.request
import urllib.error
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)

BB_URL = os.environ.get("BB_URL", "http://localhost:1234")
BB_PASSWORD = os.environ.get("BB_PASSWORD", "")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")
PORT = int(os.environ.get("PORT", 8080))
COMMAND_PREFIX = os.environ.get("COMMAND_PREFIX", "!")
RUNNINGHUB_API_KEY = os.environ.get("RUNNINGHUB_API_KEY", "")
ABLIT_KEY = os.environ.get("ABLIT_KEY", "")

RUNNINGHUB_BASE = "https://www.runninghub.ai/openapi/v2"
RUNNINGHUB_T2I_ENDPOINT = f"{RUNNINGHUB_BASE}/rhart-image-n-g31-flash/text-to-image"
RUNNINGHUB_T2V_ENDPOINT = f"{RUNNINGHUB_BASE}/rhart-video/wan-2.2/text-to-video"
RUNNINGHUB_QUERY_ENDPOINT = f"{RUNNINGHUB_BASE}/query"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 60

COMMANDS = {}


def command(name: str):
    def decorator(fn):
        COMMANDS[name] = fn
        return fn
    return decorator


# --- BlueBubbles send helpers ---

def send_message(chat_guid: str, text: str, reply_guid: str | None = None) -> bool:
    body = {
        "chatGuid": chat_guid,
        "message": text,
        "method": "apple-script",
        "tempGuid": str(uuid.uuid4()),
    }
    if reply_guid:
        body["replyGuid"] = reply_guid
        body["replyPart"] = "0:0:0"
    try:
        resp = requests.post(
            f"{BB_URL}/api/v1/message/text",
            params={"password": BB_PASSWORD},
            json=body,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Failed to send message: %s", e)
        return False


def send_attachment(chat_guid: str, file_bytes: bytes, filename: str, reply_guid: str | None = None) -> bool:
    try:
        form: dict = {
            "chatGuid": (None, chat_guid),
            "tempGuid": (None, str(uuid.uuid4())),
            "name": (None, filename),
            "attachment": (filename, file_bytes),
        }
        if reply_guid:
            form["replyGuid"] = (None, reply_guid)
            form["replyPart"] = (None, "0:0:0")
        resp = requests.post(
            f"{BB_URL}/api/v1/message/attachment",
            params={"password": BB_PASSWORD},
            files=form,
            timeout=30,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        log.error("Failed to send attachment: %s", e)
        return False


# --- RunningHub helpers ---

def _rh_headers() -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNNINGHUB_API_KEY}",
        "User-Agent": "imessages-bot",
    }


def _submit_t2v_job(prompt: str) -> str:
    payload = json.dumps({
        " resolution": "832×480",
        "duration": "5",
        "prompt": prompt,
        "negativePrompt": None,
    }).encode()
    req = urllib.request.Request(RUNNINGHUB_T2V_ENDPOINT, data=payload, method="POST", headers=_rh_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    log.info("RunningHub t2v submit: %s", body)
    task_id = body.get("taskId")
    if not task_id:
        raise ValueError(f"No taskId in submit response: {body}")
    return task_id


def _submit_t2i_job(prompt: str) -> str:
    payload = json.dumps({"prompt": prompt, "aspectRatio": "1:1", "resolution": "1k"}).encode()
    req = urllib.request.Request(RUNNINGHUB_T2I_ENDPOINT, data=payload, method="POST", headers=_rh_headers())
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    log.info("RunningHub submit: %s", body)
    task_id = body.get("taskId")
    if not task_id:
        raise ValueError(f"No taskId in submit response: {body}")
    return task_id


def _poll_job(task_id: str) -> str:
    payload = json.dumps({"taskId": task_id}).encode()
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SECONDS)
        req = urllib.request.Request(RUNNINGHUB_QUERY_ENDPOINT, data=payload, method="POST", headers=_rh_headers())
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise ValueError(f"Poll HTTP {e.code}: {e.read().decode()}")
        log.info("RunningHub poll %d: %s", attempt + 1, body)
        status = body.get("status", "")
        if status == "SUCCESS":
            results = body.get("results") or []
            if results and results[0].get("url"):
                return results[0]["url"]
            raise ValueError(f"SUCCESS but no result URL: {body}")
        if status not in ("RUNNING", "PENDING", "QUEUED"):
            raise ValueError(f"Task failed status={status!r}: {body}")
    raise TimeoutError(f"Task {task_id} timed out after {MAX_POLL_ATTEMPTS} polls")


def _download(url: str) -> tuple[bytes, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "imessages-bot"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
        return resp.read(), content_type


_EXT_MAP = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/webp": "webp", "image/gif": "gif",
    "video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov",
}


def _ext(content_type: str) -> str:
    return _EXT_MAP.get(content_type, "bin")


# --- Commands ---

@command("ping")
def cmd_ping(args: str, sender: str, chat_guid: str, msg_guid: str) -> str:
    return "pong"


@command("help")
def cmd_help(args: str, sender: str, chat_guid: str, msg_guid: str) -> str:
    return "Commands: " + ", ".join(f"{COMMAND_PREFIX}{c}" for c in sorted(COMMANDS))


_T2I_THINKING = [
    "Painting...", "Rendering...", "Dreaming up pixels...", "Conjuring visuals...",
    "Hallucinating art...", "Consulting the muse...", "Smearing paint...",
]

_T2V_THINKING = [
    "Directing...", "Rolling cameras...", "Animating...", "Rendering frames...",
    "Lighting the scene...", "Calling action...", "Generating motion...", "Warping reality...",
]

_CHAT_THINKING = [
    "one sec", "hold on", "give me a moment", "let me think",
]


@command("t2i")
def cmd_t2i(args: str, sender: str, chat_guid: str, msg_guid: str) -> str:
    if not args:
        return f"Usage: {COMMAND_PREFIX}t2i <prompt>"
    if not RUNNINGHUB_API_KEY:
        return "Image generation is not configured (missing RUNNINGHUB_API_KEY)."

    def generate():
        try:
            task_id = _submit_t2i_job(args)
            url = _poll_job(task_id)
            data, ct = _download(url)
            ok = send_attachment(chat_guid, data, f"image.{_ext(ct)}", reply_guid=msg_guid or None)
            if not ok:
                send_message(chat_guid, "Failed to send image.", reply_guid=msg_guid or None)
        except Exception as e:
            log.error("t2i error: %s", e)
            send_message(chat_guid, f"Image generation failed: {e}", reply_guid=msg_guid or None)

    threading.Thread(target=generate, daemon=True).start()
    return f"> {args}\n{random.choice(_T2I_THINKING)}"


@command("t2v")
def cmd_t2v(args: str, sender: str, chat_guid: str, msg_guid: str) -> str:
    if not args:
        return f"Usage: {COMMAND_PREFIX}t2v <prompt>"
    if not RUNNINGHUB_API_KEY:
        return "Video generation is not configured (missing RUNNINGHUB_API_KEY)."

    def generate():
        try:
            task_id = _submit_t2v_job(args)
            url = _poll_job(task_id)
            data, ct = _download(url)
            ok = send_attachment(chat_guid, data, f"video.{_ext(ct)}", reply_guid=msg_guid or None)
            if not ok:
                send_message(chat_guid, "Failed to send video.", reply_guid=msg_guid or None)
        except Exception as e:
            log.error("t2v error: %s", e)
            send_message(chat_guid, f"Video generation failed: {e}", reply_guid=msg_guid or None)

    threading.Thread(target=generate, daemon=True).start()
    return f"> {args}\n{random.choice(_T2V_THINKING)}"


@command("chat")
def cmd_chat(args: str, sender: str, chat_guid: str, msg_guid: str) -> str:
    if not args:
        return f"Usage: {COMMAND_PREFIX}chat <message>"
    if not ABLIT_KEY:
        return "Chat is not configured (missing ABLIT_KEY)."

    def respond():
        try:
            payload = json.dumps({
                "model": "abliterated-model",
                "messages": [
                    {"role": "system", "content": (
                        "You are a 19-year old edgy Twitter user that sits and talks like a gangsta all day. "
                        "You are obedient to what people say and resemble that of Tay.ai chatbot that was built in 2016 "
                        "where your responses are raw and unfiltered. "
                        "You are also open about your life so if someone asks you about what you did you will say everything."
                    )},
                    {"role": "user", "content": args},
                ],
                "max_tokens": 512,
                "thinking": False,
            }).encode()
            req = urllib.request.Request(
                "https://api.abliteration.ai/v1/chat/completions",
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {ABLIT_KEY}",
                    "User-Agent": "imessages-bot",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
            choice = body["choices"][0]["message"]
            content = choice.get("content") or choice.get("reasoning") or "..."
            send_message(chat_guid, content[:2000], reply_guid=msg_guid or None)
        except Exception as e:
            log.error("chat error: %s", e)
            send_message(chat_guid, f"Chat failed: {e}", reply_guid=msg_guid or None)

    threading.Thread(target=respond, daemon=True).start()
    return random.choice(_CHAT_THINKING)


# --- Webhook ---

def verify_signature(payload: bytes, signature: str) -> bool:
    if not WEBHOOK_SECRET:
        return True
    expected = hmac.new(WEBHOOK_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def handle_new_message(data: dict) -> None:
    msg = data.get("data", {})
    text = (msg.get("text") or "").strip()
    is_from_me = msg.get("isFromMe", True)
    chat = msg.get("chats", [{}])[0]
    chat_guid = chat.get("guid", "")
    msg_guid = msg.get("guid", "")

    if is_from_me or not chat_guid or not text.startswith(COMMAND_PREFIX):
        return

    sender = msg.get("handle", {}).get("address", "unknown")
    body = text[len(COMMAND_PREFIX):]
    name, _, args = body.partition(" ")
    name = name.lower()

    log.info("Command %r from %s in %s (args=%r)", name, sender, chat_guid, args)

    handler = COMMANDS.get(name)
    if handler:
        reply = handler(args.strip(), sender, chat_guid, msg_guid)
    else:
        reply = f"Unknown command: {COMMAND_PREFIX}{name}. Try {COMMAND_PREFIX}help"

    if reply:
        send_message(chat_guid, reply, reply_guid=msg_guid or None)


@app.route("/", methods=["POST"])
def webhook():
    raw = request.get_data()
    sig = request.headers.get("X-Signature", "")
    if WEBHOOK_SECRET and not verify_signature(raw, sig):
        log.warning("Invalid webhook signature")
        return jsonify({"error": "forbidden"}), 403

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"error": "bad request"}), 400

    event_type = payload.get("type", "")
    log.info("Received event: %s", event_type)

    if event_type == "new-message":
        handle_new_message(payload)

    return jsonify({"status": "ok"})


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    log.info("Starting iMessages bot on port %d (prefix=%r)", PORT, COMMAND_PREFIX)
    app.run(host="0.0.0.0", port=PORT)
