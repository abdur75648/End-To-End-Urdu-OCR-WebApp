import torch
import gradio as gr
from read import text_recognizer
from model import Model
from utils import CTCLabelConverter
from kraken import binarization
from kraken import pageseg as detection_model
from PIL import ImageDraw

""" vocab / character number configuration """
file = open("UrduGlyphs.txt","r",encoding="utf-8")
content = file.readlines()
content = ''.join([str(elem).strip('\n') for elem in content])
content = content+" "
""" model configuration """
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
converter = CTCLabelConverter(content)
recognition_model = Model(num_class=len(converter.character), device=device)
modrecognition_modelel = recognition_model.to(device)
recognition_model.load_state_dict(torch.load("best_norm_ED.pth", map_location=device))
recognition_model.eval()

examples = ["1.jpg","2.jpg","3.jpg"]

input = gr.Image(type="pil",image_mode="RGB", label="Input Image")

def predict(input):
    "Line Detection"
    bw_input = binarization.nlbin(input)
    bounding_boxes = detection_model.segment(bw_input)['boxes']
    bounding_boxes.sort(key=lambda x: x[1])
    
    "Draw the bounding boxes"
    draw = ImageDraw.Draw(input)
    for box in bounding_boxes:
        draw.rectangle(box, outline='red', width=3)
    
    "Crop the detected lines"
    cropped_images = []
    for box in bounding_boxes:
        cropped_images.append(input.crop(box))
    len(cropped_images)
    
    "Recognize the text"
    texts = []
    for img in cropped_images:
        texts.append(text_recognizer(img, recognition_model, converter, device))
    
    "Join the text"
    text = "\n".join(texts)
    
    "Return the image with bounding boxes and the text"
    return input,text

output_image = gr.Image(type="pil",image_mode="RGB",label="Detected Lines")
output_text = gr.Textbox(label="Recognized Text",interactive=True,show_copy_button=True)

iface = gr.Interface(predict,
                     inputs=input,
                     outputs=[output_image,output_text],
                     title="End-to-End Urdu OCR",
                     description="Demo Web App For UTRNet\n(https://github.com/abdur75648/UTRNet-High-Resolution-Urdu-Text-Recognition)",
                     examples=examples,
                     allow_flagging="never")
iface.launch()