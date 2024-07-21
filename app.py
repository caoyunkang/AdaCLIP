import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import warnings
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import json
import os
import torch
from scipy.ndimage import gaussian_filter
import cv2
from method import AdaCLIP_Trainer
import numpy as np

############ Init Model
ckt_path1 = 'weights/pretrained_mvtec_colondb.pth'
ckt_path2 = "weights/pretrained_visa_clinicdb.pth"
ckt_path3 = 'weights/pretrained_all.pth'

# Configurations
image_size = 518
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
model = "ViT-L-14-336"
prompting_depth = 4
prompting_length = 5
prompting_type = 'SD'
prompting_branch = 'VL'
use_hsf = True
k_clusters = 20

config_path = os.path.join('./model_configs', f'{model}.json')

# Prepare model
with open(config_path, 'r') as f:
    model_configs = json.load(f)

# Set up the feature hierarchy
n_layers = model_configs['vision_cfg']['layers']
substage = n_layers // 4
features_list = [substage, substage * 2, substage * 3, substage * 4]

model = AdaCLIP_Trainer(
    backbone=model,
    feat_list=features_list,
    input_dim=model_configs['vision_cfg']['width'],
    output_dim=model_configs['embed_dim'],
    learning_rate=0.,
    device=device,
    image_size=image_size,
    prompting_depth=prompting_depth,
    prompting_length=prompting_length,
    prompting_branch=prompting_branch,
    prompting_type=prompting_type,
    use_hsf=use_hsf,
    k_clusters=k_clusters
).to(device)


def process_image(image, text, options):
    # Load the model based on selected options
    if 'MVTec AD+Colondb' in options:
        model.load(ckt_path1)
    elif 'VisA+Clinicdb' in options:
        model.load(ckt_path2)
    elif 'All' in options:
        model.load(ckt_path3)
    else:
        # Default to 'All' if no valid option is provided
        model.load(ckt_path3)
        print('Invalid option. Defaulting to All.')

    # Ensure image is in RGB mode
    image = image.convert('RGB')

    # Convert PIL image to NumPy array
    np_image = np.array(image)

    # Convert RGB to BGR for OpenCV
    np_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    np_image = cv2.resize(np_image, (image_size, image_size))
    # Preprocess the image and run the model
    img_input = model.preprocess(image).unsqueeze(0)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        anomaly_map, anomaly_score = model.clip_model(img_input, [text], aggregation=True)

    # Process anomaly map
    anomaly_map = anomaly_map[0, :, :].cpu().numpy()
    anomaly_score = anomaly_score[0].cpu().numpy()
    anomaly_map = gaussian_filter(anomaly_map, sigma=4)
    anomaly_map = (anomaly_map * 255).astype(np.uint8)

    # Apply color map and blend with original image
    heat_map = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
    vis_map = cv2.addWeighted(heat_map, 0.5, np_image, 0.5, 0)

    # Convert OpenCV image back to PIL image for Gradio
    vis_map_pil = Image.fromarray(cv2.cvtColor(vis_map, cv2.COLOR_BGR2RGB))

    return vis_map_pil, f'{anomaly_score:.3f}'

# Define examples
examples = [
    ["asset/img.png", "candle", "MVTec AD+Colondb"],
    ["asset/img2.png", "bottle", "VisA+Clinicdb"],
    ["asset/img3.png", "button", "All"],
]

# Gradio interface layout
demo = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Class Name"),
        gr.Radio(["MVTec AD+Colondb",
                  "VisA+Clinicdb",
                  "All"],
        label="Pre-trained Datasets")
    ],
    outputs=[
        gr.Image(type="pil", label="Output Image"),
        gr.Textbox(label="Anomaly Score"),
    ],
    examples=examples,
    title="AdaCLIP -- Zero-shot Anomaly Detection",
    description="Upload an image, enter class name, and select pre-trained datasets to do zero-shot anomaly detection"
)

# Launch the demo
demo.launch()
# demo.launch(server_name="0.0.0.0", server_port=10002)

