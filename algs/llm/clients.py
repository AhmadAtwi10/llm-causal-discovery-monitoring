"""Access to the LLM backends used in the experiments.

Two backends are supported:
    ollama   local open-source models (llama3.3:70b, gemma3:27b, deepseek-r1:70b, mistral, mixtral:8x7b, ...).
             The Ollama server must be running and the model pulled (`ollama pull <model>`).
    openai   OpenAI models (GPT-4 Turbo in the paper). Needs the OPENAI_API_KEY environment variable.

All queries use temperature 0 and a fixed seed to make the outputs as repeatable as possible.
"""
import os

DEFAULT_MODELS = {
    "ollama": "deepseek-r1:70b",
    "openai": "gpt-4-1106-preview",  # GPT-4 Turbo
}


def require_env(name):
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set (see README, 'API keys').")
    return value


def chat(system, user, backend="ollama", model=None, num_ctx=2048, seed=1234, max_tokens=1024):
    """Send a system + user message to the chosen backend and return the answer text."""
    model = model or DEFAULT_MODELS[backend]
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if backend == "ollama":
        import ollama

        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "seed": seed, "num_ctx": num_ctx},
        )
        return response["message"]["content"].strip()

    if backend == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=require_env("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            seed=seed,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    raise ValueError(f"Unknown backend {backend!r} (expected 'ollama' or 'openai')")
