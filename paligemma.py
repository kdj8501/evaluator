import PIL
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_image(imgpath):
    image = PIL.Image.open(imgpath)
    image = np.array(image)
    # Remove alpha channel if neccessary.
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return image

def get_result_pali(imgpath, prompt, model, processor):
    image = read_image(imgpath)

    inputs = processor(text = prompt, images = image, 
                    padding = "longest", do_convert_rgb = True, return_tensors = "pt").to(DEVICE)
    inputs = inputs.to(dtype = model.dtype)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens = 128)

    result = processor.decode(output[0], skip_special_tokens = True)
    result = result[len(prompt) + 1:]
    return result