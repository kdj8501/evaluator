import openai, base64

CHATGPT_API_KEY = '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'
CHATGPT_MODEL = 'gpt-4o-mini'
# CHATGPT_MODEL = 'gpt-4o'

def encode_image(img):
    with open(img, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def get_result_chatgpt(img, prompt):
    openai.api_key = CHATGPT_API_KEY
    base64_image = encode_image(img)
    image = f"data:image/jpeg;base64,{base64_image}"
    response = openai.chat.completions.create(
        model = CHATGPT_MODEL,
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image,
                        },
                    },
                ],
            }
        ],
        max_tokens = 500
    )
    return response.choices[0].to_dict()['message']['content']
