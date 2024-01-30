# A simplified version of the original code - https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition
import math
import torch
from PIL import Image
import torch.utils.data
from utils import NormalizePAD

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def text_recognizer(img_cropped, model, converter, device):
    """ Image processing """
    img = img_cropped.convert('L')
    img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    w, h = img.size
    ratio = w / float(h)
    if math.ceil(32 * ratio) > 400:
        resized_w = 400
    else:
        resized_w = math.ceil(32 * ratio)
    img = img.resize((resized_w, 32), Image.Resampling.BICUBIC)
    transform = NormalizePAD((1, 32, 400))
    img = transform(img)
    img = img.unsqueeze(0)
    batch_size = 1
    img = img.to(device)
    
    """ Prediction """
    preds = model(img)
    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
    _, preds_index = preds.max(2)
    preds_str = converter.decode(preds_index.data, preds_size.data)[0]
    return preds_str

# if __name__ == '__main__':
#     image_path = "test.jpg"
#     img_cropped = Image.open(image_path)
#     preds_str = text_recognizer(img_cropped)
#     print(preds_str)