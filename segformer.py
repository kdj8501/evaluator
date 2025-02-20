import torch, cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

CITYSCAPES_COLORMAP = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
    [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
    [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
    [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype = np.uint8)

def get_result_seg(img, model, processor, device):
    if (type(img) == np.ndarray):
        image = Image.fromarray(img).convert('RGB')
    else:
        image = Image.open(img).convert('RGB')
    inputs = processor(images = image, return_tensors = 'pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    segmentation_map = torch.argmax(logits, dim = 1).squeeze().cpu().numpy()
    segmentation_colored = np.zeros((*segmentation_map.shape, 3), dtype = np.uint8)

    for class_id in range(len(CITYSCAPES_COLORMAP)):
        segmentation_colored[segmentation_map  == class_id] = CITYSCAPES_COLORMAP[class_id]

    return segmentation_colored