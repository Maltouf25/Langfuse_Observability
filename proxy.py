# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
from langfuse import Langfuse

# ==== Hardcode your keys here (⚠️ not recommended for prod) ====

MISTRAL_API_KEY      = "IJtcxcjoTgYbolJzCh4gz2xbmiPB9QW8"
MISTRAL_MODEL        = "mistral-small-latest"   # or another model
LF_PUBLIC_KEY  = "pk-lf-a104f5c6-5aaa-431e-837f-81a4d5adb44a"
LF_SECRET_KEY   = "sk-lf-15ec75b4-736d-4126-9ab1-195512e29c27"
LF_HOST    = "https://langfuse.libeo.tech" 
# ===============================================================

lf = Langfuse(public_key=LF_PUBLIC_KEY, secret_key=LF_SECRET_KEY, host=LF_HOST)
app = FastAPI()

class TraceIn(BaseModel):
    user_id: Optional[str] = None
    tags: Optional[list[str]] = []

class ObsIn(BaseModel):
    trace_id: str
    name: str
    model: Optional[str] = None
    input: Dict[str, Any]
    output: Optional[Any] = None
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    as_type: str = "generation"

@app.post("/trace")
def new_trace(body: TraceIn):
    with lf.start_as_current_span(name="n8n-run", input={"source":"n8n"}) as root:
        root.update_trace(user_id=body.user_id, tags=body.tags or [])
        return {"trace_id": root.id}

@app.post("/observation")
def add_observation(body: ObsIn):
    with lf.start_as_current_span(trace_id=body.trace_id, name="n8n-bridge"):
        with lf.start_as_current_observation(
            name=body.name,
            as_type=body.as_type,
            model=body.model,
            input=body.input,
            metadata=body.metadata or {},
        ) as obs:
            u = body.usage or {}
            obs.update(
                output=body.output,
                usage_details={
                    "input_tokens": u.get("prompt_tokens"),
                    "output_tokens": u.get("completion_tokens"),
                },
            )
    lf.flush()
    return {"ok": True}
