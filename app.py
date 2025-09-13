# app.py (hardened)
import os, time, requests
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

# --- Langfuse optional init (do not crash app if misconfigured) ---
LF_ENABLED = True
try:
    from langfuse import Langfuse
    LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
    LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_HOST       = os.getenv("LANGFUSE_HOST")
    if not (LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY):
        LF_ENABLED = False
    else:
        lf = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)
except Exception:
    LF_ENABLED = False
    lf = None  # type: ignore

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
PROXY_TOKEN     = os.getenv("PROXY_TOKEN")  # optional shared secret

if not MISTRAL_API_KEY:
    raise RuntimeError("Missing MISTRAL_API_KEY")

app = FastAPI(title="n8n → Mistral proxy with Langfuse (safe)")

@app.get("/")
def root():
    return {"service": "langfuse-mistral-proxy", "endpoints": ["/health", "/mistral/chat/completions"], "langfuse_enabled": LF_ENABLED}

@app.get("/health")
def health():
    return {"ok": True, "langfuse_enabled": LF_ENABLED}

def call_mistral(body: dict) -> tuple[dict, int]:
    """Single place to call Mistral; returns (data, status_code)."""
    resp = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"},
        json=body,
        timeout=60,
    )
    try:
        data = resp.json()
    except Exception:
        # If upstream ever returns non-JSON, still surface the raw text
        data = {"error": {"status": resp.status_code, "text": resp.text}}
    return data, resp.status_code

@app.post("/mistral/chat/completions")
async def mistral_chat(request: Request, x_proxy_token: str | None = Header(None)):
    # Gate with shared secret if configured
    if PROXY_TOKEN and x_proxy_token != PROXY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Parse body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = (body or {}).get("model", "unknown")
    session_id = request.headers.get("x-lf-session-id")

    # If Langfuse is off/unreachable, just forward
    if not LF_ENABLED:
        data, code = call_mistral(body)
        return JSONResponse(data, status_code=code)

    # Otherwise try to instrument, but never fail the request if LF breaks
    try:
        with lf.start_as_current_span(
            name="n8n-run",
            input={"entry": "mistral.chat.completions"},
            metadata={"source": "n8n"},
        ) as root:
            if session_id:
                root.update_trace(session_id=session_id, tags=["n8n", "mistral"])

            t0 = time.time()
            with lf.start_as_current_observation(
                name="Extract BL rows with Mistral",
                as_type="generation",
                model=model,
                input=body,
                metadata={"node": "Extract BL rows with Mistral"},
            ) as obs:
                data, code = call_mistral(body)

                # Safely extract output/usage (works even on errors)
                try:
                    msg = (data.get("choices", [{}])[0].get("message") or {})
                except Exception:
                    msg = {}
                content = msg.get("content")
                usage = data.get("usage") or {}

                obs.update(
                    output={"content": content, "raw": data},
                    usage_details={
                        "input_tokens": usage.get("prompt_tokens"),
                        "output_tokens": usage.get("completion_tokens"),
                    },
                    metadata={
                        "httpStatus": code,
                        "duration_ms": int((time.time() - t0) * 1000),
                    },
                )
        try:
            lf.flush()
        except Exception:
            # Never break response if flush fails
            pass

        return JSONResponse(data, status_code=code)

    except Exception as e:
        # If Langfuse path fails for any reason, fallback to plain forward
        data, code = call_mistral(body)
        # attach minimal hint for debugging without leaking secrets
        data.setdefault("_proxy_note", "LF instrumentation disabled due to error")
        return JSONResponse(data, status_code=code)
