import os
from concurrent.futures import ThreadPoolExecutor


def client(base_url, api_key_env="OPENROUTER_API_KEY"):
    # Imported lazily so the pure parsing and scoring code stays importable
    # without the API dependency -- which is also what makes it unit-testable.
    from openai import OpenAI

    # A local vLLM server ignores the key; "EMPTY" is the conventional placeholder.
    return OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"), max_retries=5)


def complete(llm, model, prompt, max_tokens=1024, temperature=0.7, extra=None):
    resp = llm.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra_body=extra,
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError(f"null content from {model} (finish_reason="
                           f"{resp.choices[0].finish_reason})")
    return content


def pmap(fn, items, workers=8, desc=None):
    """Parallel map, order-preserving. With desc, shows a progress bar --
    tqdm ships with huggingface_hub, so it is already everywhere this runs."""
    items = list(items)
    with ThreadPoolExecutor(workers) as pool:
        results = pool.map(fn, items)
        if desc:
            from tqdm import tqdm

            results = tqdm(results, total=len(items), desc=f"  {desc}", leave=False)
        return list(results)


def load_local(model_id):
    """Load a model for forward passes. Imported lazily so the API-only paths do
    not pay for torch."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    ).eval()
    return tok, model, device
