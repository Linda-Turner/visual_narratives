import json, sys
import requests
import filetype

from generate_description.utils import encode_image


def describe(tweet_text, image_path, prompt, credentials, max_tokens):
    kind = filetype.guess(image_path)
    if kind is None:
        return None
    base64_image = encode_image(image_path)
    payload = {
        "model": "gpt-4.1-nano-2025-04-14",
        "messages": [
            {
                "role":"system",
                "content": prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": tweet_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{kind.mime};base64,{base64_image}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.01
    }
    result = gpt4(payload, credentials)
    return result

def narrative(description, prompt, credentials, max_tokens):
    payload = {
        "model": "gpt-4.1-nano-2025-04-14",
        "messages": [
            {
                "role": "assistant",
                "content": description
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.01
    }
    result = gpt4(payload, credentials)
    return result


def gpt4(payload, cred):
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=cred, json=payload)
    data = response.json()
    if 'error' in data:
        sys.stderr.write(json.dumps(data) + "\n")
        if 'code' in data['error']:
            if data['error']['code'] == 'sanitizer_server_error':
                return '[wreq]'
        return '[err]'
    if 'message' in data['choices'][0]:
        result = data['choices'][0]['message']['content']
        return result
    else:
        return "no result"