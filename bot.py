import os
import hmac
import hashlib
import json
import logging
import random
import time
import threading
import uuid
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import requests
import boto3
from botocore.exceptions import ClientError

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
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET = os.environ.get("S3_BUCKET", "")

RUNNINGHUB_BASE = "https://www.runninghub.ai/openapi/v2"
RUNNINGHUB_T2I_ENDPOINT = f"{RUNNINGHUB_BASE}/rhart-image-n-g31-flash/text-to-image"
RUNNINGHUB_T2V_ENDPOINT = f"{RUNNINGHUB_BASE}/rhart-video/wan-2.2/text-to-video"
RUNNINGHUB_I2I_ENDPOINT = f"{RUNNINGHUB_BASE}/bytedance/jimeng-4.6/image-to-image"
RUNNINGHUB_I2V_ENDPOINT = f"{RUNNINGHUB_BASE}/rhart-video-g-official/image-to-video"
RUNNINGHUB_QUERY_ENDPOINT = f"{RUNNINGHUB_BASE}/query"
POLL_INTERVAL_SECONDS = 5
MAX_POLL_ATTEMPTS = 60

_SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}

_EXT_MAP = {
    "image/png": "png", "image/jpeg": "jpg", "image/jpg": "jpg",
    "image/webp": "webp", "image/gif": "gif",
    "video/mp4": "mp4", "video/webm": "webm", "video/quicktime": "mov",
}

_THINKING = {
    "t2i": ["Painting...", "Rendering...", "Dreaming up pixels...", "Conjuring visuals...",
            "Hallucinating art...", "Consulting the muse...", "Smearing paint..."],
    "t2v": ["Directing...", "Rolling cameras...", "Animating...", "Rendering frames...",
            "Lighting the scene...", "Calling action...", "Generating motion...", "Warping reality..."],
    "i2i": ["Transforming...", "Remixing...", "Reimagining...", "Morphing...", "Reshaping..."],
    "i2v": ["Animating...", "Bringing to life...", "Rolling film...", "Generating motion..."],
    "chat": ["one sec", "hold on", "give me a moment", "let me think"],
}

COMMANDS = {}
_s3 = None


def command(name: str):
    def decorator(fn):
        COMMANDS[name] = fn
        return fn
    return decorator


def _get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3", region_name=AWS_REGION)
    return _s3


def _ext(content_type: str) -> str:
    return _EXT_MAP.get(content_type, "bin")


# --- BlueBubbles helpers ---

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
    form: dict = {
        "chatGuid": (None, chat_guid),
        "tempGuid": (None, str(uuid.uuid4())),
        "name": (None, filename),
        "attachment": (filename, file_bytes),
    }
    if reply_guid:
        form["replyGuid"] = (None, reply_guid)
        form["replyPart"] = (None, "0:0:0")
    try:
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


# --- S3 helpers ---

def _upload_to_s3(data: bytes, content_type: str) -> tuple[str, str]:
    key = f"imessages-bot-tmp/{uuid.uuid4().hex}"
    s3 = _get_s3()
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)
    presigned_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=3600,
    )
    log.info("Uploaded to S3: %s", key)
    return presigned_url, key


def _delete_from_s3(key: str) -> None:
    try:
        _get_s3().delete_object(Bucket=S3_BUCKET, Key=key)
        log.info("Deleted S3 object: %s", key)
    except ClientError as e:
        log.error("Failed to delete S3 object %s: %s", key, e)


def _validate_image_attachment(attachments: list) -> tuple[dict, str] | tuple[None, str]:
    if not attachments:
        return None, "Please attach an image to the message."
    att = attachments[0]
    mime = (att.get("mimeType") or "").lower()
    if not mime.startswith("image/"):
        return None, f"Attachment is not an image (got {mime or 'unknown type'}). Please attach a photo."
    if mime not in _SUPPORTED_IMAGE_TYPES:
        return None, f"Unsupported image type: {mime}. Supported: PNG, JPEG, WEBP, GIF."
    return att, ""


def _stage_attachment(att: dict) -> tuple[str, str]:
    guid = att.get("guid", "")
    if not guid:
        raise ValueError("Attachment has no guid")
    resp = requests.get(
        f"{BB_URL}/api/v1/attachment/{guid}/download",
        params={"password": BB_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    if not content_type.startswith("image/"):
        raise ValueError(f"Downloaded content is not an image: {content_type}")
    return _upload_to_s3(resp.content, content_type)


# --- RunningHub helpers ---

def _rh_post(endpoint: str, body: dict) -> str:
    resp = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {RUNNINGHUB_API_KEY}",
            "User-Agent": "imessages-bot",
        },
        json=body,
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    log.info("RunningHub submit %s: %s", endpoint, result)
    task_id = result.get("taskId")
    if not task_id:
        raise ValueError(f"No taskId in response: {result}")
    return task_id


def _poll_job(task_id: str) -> str:
    for attempt in range(MAX_POLL_ATTEMPTS):
        time.sleep(POLL_INTERVAL_SECONDS)
        resp = requests.post(
            RUNNINGHUB_QUERY_ENDPOINT,
            headers={"Authorization": f"Bearer {RUNNINGHUB_API_KEY}", "User-Agent": "imessages-bot"},
            json={"taskId": task_id},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
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
    resp = requests.get(url, headers={"User-Agent": "imessages-bot"}, timeout=30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/png").split(";")[0].strip()
    return resp.content, content_type


def _run_generation(submit_fn, chat_guid: str, msg_guid: str, label: str, s3_key: str | None = None):
    def generate():
        url = None
        try:
            task_id = submit_fn()
            url = _poll_job(task_id)
        except Exception as e:
            log.error("%s error: %s", label, e)
            send_message(chat_guid, f"{label} failed: {e}", reply_guid=msg_guid or None)
        finally:
            if s3_key:
                _delete_from_s3(s3_key)
        if url is None:
            return
        try:
            data, ct = _download(url)
            ok = send_attachment(chat_guid, data, f"output.{_ext(ct)}", reply_guid=msg_guid or None)
            if not ok:
                send_message(chat_guid, f"Failed to send {label} result.", reply_guid=msg_guid or None)
        except Exception as e:
            log.error("%s send error: %s", label, e)
            send_message(chat_guid, f"Failed to send result: {e}", reply_guid=msg_guid or None)

    threading.Thread(target=generate, daemon=True).start()


# --- Commands ---

@command("ping")
def cmd_ping(args, sender, chat_guid, msg_guid, attachments=None):
    return "pong"


@command("help")
def cmd_help(args, sender, chat_guid, msg_guid, attachments=None):
    return "Commands: " + ", ".join(f"{COMMAND_PREFIX}{c}" for c in sorted(COMMANDS))


@command("t2i")
def cmd_t2i(args, sender, chat_guid, msg_guid, attachments=None):
    if not args:
        return f"Usage: {COMMAND_PREFIX}t2i <prompt>"
    if not RUNNINGHUB_API_KEY:
        return "Image generation is not configured (missing RUNNINGHUB_API_KEY)."
    _run_generation(
        lambda: _rh_post(RUNNINGHUB_T2I_ENDPOINT, {"prompt": args, "aspectRatio": "1:1", "resolution": "1k"}),
        chat_guid, msg_guid, "Image generation",
    )
    return f"> {args}\n{random.choice(_THINKING['t2i'])}"


@command("t2v")
def cmd_t2v(args, sender, chat_guid, msg_guid, attachments=None):
    if not args:
        return f"Usage: {COMMAND_PREFIX}t2v <prompt>"
    if not RUNNINGHUB_API_KEY:
        return "Video generation is not configured (missing RUNNINGHUB_API_KEY)."
    _run_generation(
        lambda: _rh_post(RUNNINGHUB_T2V_ENDPOINT, {" resolution": "832×480", "duration": "5", "prompt": args, "negativePrompt": None}),
        chat_guid, msg_guid, "Video generation",
    )
    return f"> {args}\n{random.choice(_THINKING['t2v'])}"


@command("i2i")
def cmd_i2i(args, sender, chat_guid, msg_guid, attachments=None):
    if not args:
        return f"Usage: {COMMAND_PREFIX}i2i <prompt>  (attach an image to the message)"
    if not RUNNINGHUB_API_KEY:
        return "Image transformation is not configured (missing RUNNINGHUB_API_KEY)."
    if not S3_BUCKET:
        return "Image transformation is not configured (missing S3_BUCKET)."
    att, err = _validate_image_attachment(attachments or [])
    if not att:
        return err
    try:
        image_url, s3_key = _stage_attachment(att)
    except Exception as e:
        log.error("i2i stage error: %s", e)
        return f"Failed to prepare image: {e}"
    _run_generation(
        lambda: _rh_post(RUNNINGHUB_I2I_ENDPOINT, {"prompt": args, "imageUrls": [image_url]}),
        chat_guid, msg_guid, "Image transformation", s3_key=s3_key,
    )
    return f"> {args}\n{random.choice(_THINKING['i2i'])}"


@command("i2v")
def cmd_i2v(args, sender, chat_guid, msg_guid, attachments=None):
    if not args:
        return f"Usage: {COMMAND_PREFIX}i2v <prompt>  (attach an image to the message)"
    if not RUNNINGHUB_API_KEY:
        return "Video generation is not configured (missing RUNNINGHUB_API_KEY)."
    if not S3_BUCKET:
        return "Video generation is not configured (missing S3_BUCKET)."
    att, err = _validate_image_attachment(attachments or [])
    if not att:
        return err
    try:
        image_url, s3_key = _stage_attachment(att)
    except Exception as e:
        log.error("i2v stage error: %s", e)
        return f"Failed to prepare image: {e}"
    _run_generation(
        lambda: _rh_post(RUNNINGHUB_I2V_ENDPOINT, {"prompt": args, "imageUrl": image_url, "resolution": "720p", "duration": "6"}),
        chat_guid, msg_guid, "Video generation", s3_key=s3_key,
    )
    return f"> {args}\n{random.choice(_THINKING['i2v'])}"


@command("chat")
def cmd_chat(args, sender, chat_guid, msg_guid, attachments=None):
    if not args:
        return f"Usage: {COMMAND_PREFIX}chat <message>"
    if not ABLIT_KEY:
        return "Chat is not configured (missing ABLIT_KEY)."

    def respond():
        try:
            resp = requests.post(
                "https://api.abliteration.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {ABLIT_KEY}", "User-Agent": "imessages-bot"},
                json={
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
                },
                timeout=60,
            )
            resp.raise_for_status()
            choice = resp.json()["choices"][0]["message"]
            content = choice.get("content") or choice.get("reasoning") or "..."
            send_message(chat_guid, content[:2000], reply_guid=msg_guid or None)
        except Exception as e:
            log.error("chat error: %s", e)
            send_message(chat_guid, f"Chat failed: {e}", reply_guid=msg_guid or None)

    threading.Thread(target=respond, daemon=True).start()
    return random.choice(_THINKING["chat"])


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
    attachments = msg.get("attachments", [])

    if is_from_me or not chat_guid or not text.startswith(COMMAND_PREFIX):
        return

    sender = msg.get("handle", {}).get("address", "unknown")
    body = text[len(COMMAND_PREFIX):]
    name, _, args = body.partition(" ")
    name = name.lower()

    log.info("Command %r from %s in %s (args=%r)", name, sender, chat_guid, args)

    handler = COMMANDS.get(name)
    if handler:
        reply = handler(args.strip(), sender, chat_guid, msg_guid, attachments)
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
