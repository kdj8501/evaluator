from google import genai
from PIL import Image

GOOGLE_API_KEY = 'AIzaSyCpDGzYFds5xLHvSGjuU5lCudCjDD1tz8M'
GOOGLE_API_MODEL = 'gemini-2.0-flash'

def get_result_gemini(imgpath, prompt):
    client = genai.Client(api_key = GOOGLE_API_KEY)
    image = Image.open(imgpath)
    response = client.models.generate_content(
        model = GOOGLE_API_MODEL,
        contents = [prompt, image]
    )
    return response.text

# path = 'C:/Users/koo/workspace/dataset/20yolo'
# prompt = "Is this car or truck? Answer just car or truck in one word."
# file = path + '/19.png'
# print(get_result_gemini(file, prompt))