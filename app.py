import os, time, uuid, requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv()

MISTRAL_API_KEY       = os.getenv("MISTRAL_API_KEY")
LANGFUSE_PUBLIC_KEY   = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY   = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST         = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not all([MISTRAL_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY]):
    raise RuntimeError("Missing one of: MISTRAL_API_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY")

app = FastAPI(title="n8n → Mistral proxy with Langfuse")
lf = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LANGFUSE_HOST)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/mistral/chat/completions")
async def mistral_chat(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = body.get("model", "unknown-model")
    session_id = request.headers.get("x-lf-session-id")  # optional, we’ll set this from n8n later

    # Root span (trace) for this n8n run
    with lf.start_as_current_span(
        name="n8n-run",
        input={"entry": "mistral.chat.completions"},
        metadata={"source": "n8n"},
    ) as root:
        if session_id:
            root.update_trace(session_id=session_id, tags=["n8n", "mistral"])

        t0 = time.time()
        # Observation for the actual LLM call
        with lf.start_as_current_observation(
            name="Extract BL rows with Mistral",
            as_type="generation",
            model=model,
            input=body,
            metadata={"node": "Extract BL rows with Mistral"},
        ) as obs:
            try:
                resp = requests.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=60,
                )
                data = resp.json()
            except Exception as e:
                obs.update(metadata={"proxy_error": str(e)})
                lf.flush()
                raise HTTPException(status_code=502, detail=f"Upstream call failed: {e}")

            msg = (data.get("choices", [{}])[0].get("message") or {})
            content = msg.get("content")
            usage = data.get("usage") or {}
            duration_ms = int((time.time() - t0) * 1000)

            obs.update(
                output={"content": content, "raw": data},
                usage_details={
                    "input_tokens": usage.get("prompt_tokens"),
                    "output_tokens": usage.get("completion_tokens"),
                },
                metadata={"httpStatus": resp.status_code, "duration_ms": duration_ms},
            )

    lf.flush()
    return JSONResponse(data, status_code=resp.status_code)
