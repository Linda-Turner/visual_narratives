from transformers import AutoProcessor, AutoModelForMultimodalLM
import filetype

from generate_description.utils import encode_image, get_device


def load_gemma_model(gemma_description_model: str):
    """
    Load a pre-trained GEMMA multimodal model and its processor.

    Args:
        gemma_description_model (str): Hugging Face model identifier for the
            GEMMA multimodal model.

    Returns:
        tuple: A tuple containing the loaded model, processor, and device.
    """
    print("Loading GEMMA model...")

    device = get_device()
    model = AutoModelForMultimodalLM.from_pretrained(gemma_description_model, dtype="auto")
    model.to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(gemma_description_model)
    return model, processor, device


def description(
        tweet_text: str,
        image_path: str,
        prompt: str,
        processor,
        model,
        device,
        max_tokens: int
    ):
    """
    Generate a description of an image using the GEMMA model.

    Args:
        image_path (str): Path to the image to describe.
        prompt (str): System prompt containing the instructions for generating
            the description.
        processor: GEMMA processor used to prepare the input for the model.
        model: Loaded GEMMA multimodal model.
        device: Device on which the model is running.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated image description.
    """
    image_path = image_path
    kind = filetype.guess(image_path)
    base64_image = encode_image(image_path)
    messages = [
    {
        "role": "system",
        "content": prompt
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": f"data:{kind.mime};base64,{base64_image}"
            },
            {
                "type": "text",
                "text": str(tweet_text)
            }
        ]
    }
    ]

    result = gemma(messages,processor,model,device,max_tokens)
    return result


def narrative(
        description: str,
        prompt: str,
        processor,
        model,
        device,
        max_tokens: int
    ):
    """
    Generate a narrative based on a previously generated image description.

    Args:
        description (str): Previously generated description of the image.
        prompt (str): User prompt containing the instructions for generating
            the narrative.
        processor: GEMMA processor used to prepare the input for the model.
        model: Loaded GEMMA multimodal model.
        device: Device on which the model is running.
        max_tokens (int): Maximum number of tokens to generate.

    Returns:
        str: The generated narrative.
    """
    messages_step2 = [
    {
        "role": "assistant",
        "content": description
    },
    {
        "role": "user",
        "content": prompt
    }
    ]
    result = gemma(messages_step2,processor,model,device,max_tokens)
    return result


def gemma(
        messages,
        processor,
        model,
        device,
        max_new_tokens: int
    ):
    """
    Generate a response using the GEMMA model.

    Args:
        messages: Chat messages containing the system, user, or assistant
            instructions and content.
        processor: GEMMA processor used to tokenize and prepare the messages.
        model: Loaded GEMMA multimodal model.
        device: Device on which the model is running.
        max_new_tokens (int): Maximum number of new tokens to generate.

    Returns:
        str: The generated response from the model.
    """
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(device)

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    response = processor.decode(outputs[0][input_len:],skip_special_tokens=True)
    response = response.replace("\n", " ").replace("\r", "")
    return response.strip()