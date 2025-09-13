import os, requests
from langfuse import Langfuse
import requests 
import uuid
from langfuse import Langfuse

# ==== PUT YOUR KEYS HERE (for a quick local test only) ====
MISTRAL_API_KEY      = "IJtcxcjoTgYbolJzCh4gz2xbmiPB9QW8"
MISTRAL_MODEL        = "mistral-small-latest"   # or another model
LANGFUSE_PUBLIC_KEY  = "pk-lf-a104f5c6-5aaa-431e-837f-81a4d5adb44a"
LANGFUSE_SECRET_KEY  = "sk-lf-15ec75b4-736d-4126-9ab1-195512e29c27"
LF_HOST    = "https://langfuse.libeo.tech"  # keep default if on Cloud
# ==========================================================

# Init Langfuse
langfuse = Langfuse(public_key=LANGFUSE_PUBLIC_KEY, secret_key=LANGFUSE_SECRET_KEY, host=LF_HOST)
prompt = langfuse.get_prompt("prompt_test")
prompt_text = prompt.compile() 
# 2) Define a tiny prompt
messages = [
    {"role": "system", "content": "Tu es un ingénieur backend concis et précis."},
    {"role": "user", "content": prompt_text}
]

run_id = str(uuid.uuid4())
content = None

# Root span = the whole test run (the trace)
with langfuse.start_as_current_span(name="smoke-test", input={"env": "local", "run_id": run_id}) as root:
    root.update_trace(user_id="tester_1", tags=["smoke", "mistral"])

    # ✅ v3 API (no deprecation): observation as a generation
    with langfuse.start_as_current_observation(
        name="qa-mistral",
        as_type="generation",
        model=MISTRAL_MODEL,
        input={"messages": messages},
        metadata={"topic": "webhooks_vs_http", "run_id": run_id},
    ) as obs:

        # Call Mistral
        resp = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"model": MISTRAL_MODEL, "messages": messages},
            timeout=30,
        )
        data = resp.json()

        # Update Langfuse with output + usage
        content = (data.get("choices", [{}])[0]
                      .get("message", {})
                      .get("content"))
        usage = data.get("usage") or {}

        obs.update(
            output=content,
            usage_details={
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
            },
            metadata={"httpStatus": resp.status_code},
        )

# Ensure telemetry is sent
langfuse.flush()

print(content or data)