import os


def call_litellm(
    prompt: str,
    max_tokens: int,
    temperature: float,
    system_prompt: str,
    model: str,
) -> tuple[str, int, int]:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.environ["LITELLM_BASE_URL"],
        api_key=os.environ["LITELLM_API_KEY"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    usage = response.usage
    return (
        response.choices[0].message.content or "",
        int(usage.prompt_tokens if usage else 0),
        int(usage.completion_tokens if usage else 0),
    )