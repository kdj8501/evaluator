import google.generativeai as genai
from PIL import Image

GOOGLE_API_KEY = 'AIzaSyCB-bkTNh67nGfAV7vxwodcqbHLPIYXhZs'
GOOGLE_API_MODEL = 'gemini-1.5-flash'

def get_result_gemini(imgpath, prompt):
    genai.configure(api_key = GOOGLE_API_KEY)
    model = genai.GenerativeModel(GOOGLE_API_MODEL)
    image = Image.open(imgpath)
    response = model.generate_content([prompt, image])
    return response.text