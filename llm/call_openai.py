from openai import AzureOpenAI
from azure.identity import AzureCliCredential, get_bearer_token_provider
import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

azure_endpoint = os.getenv("AZURE_ENDPOINT") or os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")
auth_mode = os.getenv("AZURE_AUTH_MODE", "api_key").lower()


def create_client():
    client_config = {
        "api_version": api_version,
        "azure_endpoint": azure_endpoint,
    }
    if auth_mode == "azure_cli":
        client_config["azure_ad_token_provider"] = get_bearer_token_provider(
            AzureCliCredential(),
            "https://cognitiveservices.azure.com/.default",
        )
    else:
        client_config["api_key"] = api_key
    return AzureOpenAI(**client_config)


client = create_client()


def build_messages(prompt, system_prompt):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_openai(prompt, 
             max_tokens = 200, 
             temperature = 0, 
             system_prompt = None,
             model = deployment_name):
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=build_messages(prompt, system_prompt),
    )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    response_msg = response.choices[0].message.content
    return response_msg, input_tokens, output_tokens

def call_openai_gpt5_models(prompt, 
             max_completion_tokens = 200, 
             temperature = 0, 
             system_prompt = None,
             model = deployment_name,
             reasoning_effort = "minimal",
             ):
    
    if model == "gpt-5-chat":
        response = client.chat.completions.create(
                model=model,
                max_completion_tokens=max_completion_tokens,
                messages=build_messages(prompt, system_prompt),
            )
    else:
        response = client.chat.completions.create(
            model=model,
            reasoning_effort=reasoning_effort,
            max_completion_tokens=max_completion_tokens,
            messages=build_messages(prompt, system_prompt),
        )
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    response_msg = response.choices[0].message.content
    return response_msg, input_tokens, output_tokens

if __name__ == "__main__":
    prompt = """
                rephrase the following user utterance, leaving the variables. give 10 examples:
                'bitte zur <BMW_App> <NP_appSynonyms> gehen'
            """
    print(call_openai(prompt=prompt))