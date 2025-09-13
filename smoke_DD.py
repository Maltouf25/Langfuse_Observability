import os
os.environ["DD_TRACE_ENABLED"] = "false"
os.environ["DD_PROFILING_ENABLED"] = "false"
os.environ["DD_TRACE_DEBUG"] = "true"
os.environ["DD_SERVICE"] = "smoke-langfuse"
os.environ["DD_TELEMETRY_ENABLED"] = "false"


import uuid
import json
import requests
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import workflow, llm




# Initialize Datadog LLM Observability
DD_API_KEY = "6fa027d10d440a8578b29f4990096258"
DD_SITE = "datadoghq.eu"           # or "datadoghq.com"
MISTRAL_API_KEY = "IJtcxcjoTgYbolJzCh4gz2xbmiPB9QW8"
MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

LLMObs.enable(
    api_key=DD_API_KEY,
    site=DD_SITE,
    ml_app="mistralapp",
    # ml_app_env="dev",
    # ml_app_version="0.1.0",
)

MISTRAL_MODEL = "mistral-small-latest"
MISTRAL_URL   = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

@llm(model_name=MISTRAL_MODEL, model_provider="mistral", name="call_mistral")
def call_mistral(messages):
    resp = requests.post(MISTRAL_URL, headers=HEADERS,
                         json={"model": MISTRAL_MODEL, "messages": messages},
                         timeout=30)
    data = resp.json()
    usage = data.get("usage") or {}
    # annotate current LLM span so it's searchable
    LLMObs.annotate(
        metrics={"input_tokens": usage.get("prompt_tokens"),
                 "output_tokens": usage.get("completion_tokens")},
        tags={"http.status_code": str(resp.status_code),
              "debug.marker": "llmobs-smoke"},
    )
    return data

@workflow(name="mistral-smoke-workflow", session_id="tester_1")
def run_smoke():
    msgs = [
        {"role": "system", "content": "Tu es un ingénieur backend concis et précis."},
        {"role": "user",   "content": "Dis bonjour en une seule phrase."},
    ]
    result = call_mistral(msgs)
    content = (result.get("choices", [{}])[0].get("message", {}).get("content"))
    print(content or json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    run_smoke()
    LLMObs.flush()