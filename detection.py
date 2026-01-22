from ultralytics import YOLO
from segment_anything import build_sam, SamAutomaticMaskGenerator, SamPredictor
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from transformers import CLIPProcessor, CLIPModel

import gradio as gr

_model_loader_instance = None

from huggingface_hub import hf_hub_download

sam_vit_h_4b8939 = hf_hub_download(repo_id="HCMUE-Research/SAM-vit-h", filename="sam_vit_h_4b8939.pth")
yolov8x = hf_hub_download(repo_id="Ultralytics/YOLOv8", filename="yolov8x.pt")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mask_predictor = SamPredictor(
#     build_sam(checkpoint="sam_vit_h_4b8939.pth").to(device)
# )

# model = YOLO("yolov8x.pt")

# if torch.cuda.is_available():
#     model.to("cuda")

# clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clipmodel.to(device)


class ModelLoader:
    def __init__(self):
        # Prevent re-initialization if the instance already exists
        if hasattr(self, 'models_loaded') and self.models_loaded:
            return

        print("--- LOADING ALL AI MODELS (This should happen only once) ---")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SAM
        self.mask_predictor = SamPredictor(
            build_sam(checkpoint=sam_vit_h_4b8939).to(self.device)
        )
        
        # YOLO
        self.yolo_model = YOLO(yolov8x)
        if torch.cuda.is_available():
            self.yolo_model.to("cuda")

        # CLIP
        self.clipmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clipmodel.to(self.device)

        self.models_loaded = True
        print("--- ALL MODELS LOADED ---")

    # 2. Implement the Singleton method
    @classmethod
    def get_instance(cls):
        global _model_loader_instance
        if _model_loader_instance is None:
            _model_loader_instance = cls()
        return _model_loader_instance
    

@torch.no_grad()
def retriev(elements):
    loader = ModelLoader.get_instance()
    preprocessed_images = loader.processor(images=elements, return_tensors="pt")
    tokenized_text = loader.processor(text=["woman"], padding=True, return_tensors="pt")

    preprocessed_images["pixel_values"] = preprocessed_images["pixel_values"].to(loader.device)
    tokenized_text["input_ids"] = tokenized_text["input_ids"].to(loader.device)
    tokenized_text["attention_mask"] = tokenized_text["attention_mask"].to(loader.device)

    image_features = loader.clipmodel.get_image_features(**preprocessed_images)
    text_features = loader.clipmodel.get_text_features(**tokenized_text)

    # for text in text_features:
    #     print(text)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs = 100.0 * image_features @ text_features.T
    return probs[:, 0].softmax(dim=0)


def get_indexes_of_values_above_threshold(values, threshold):
    return [i for i, v in enumerate(values) if v > threshold]


def segment(
    clip_threshold: float,
    image_path: str,
):
    loader = ModelLoader.get_instance()
    results = loader.yolo_model.predict(image_path)

    result = results[0]

    boxes = result.boxes

    person_boxes = []

    cropped_boxes = []

    image_bgr = cv2.imread(image_path)

    imageToCrop = Image.open(image_path)

    for box in result.boxes:
        if result.names[box.cls[0].item()] == "person":
            cords = box.xyxy[0].tolist()
            cords = np.array(cords)

            crop_image = imageToCrop.crop(
                (int(cords[0]), int(cords[1]), int(cords[2]), int(cords[3]))
            )

            cropped_boxes.append({"cords": cords, "img": crop_image})


    scores = retriev([cb["img"] for cb in cropped_boxes])

    print(scores)

    indexes = get_indexes_of_values_above_threshold(scores, clip_threshold)

    print(indexes)

    womenBoxes = [cropped_boxes[i] for i in indexes]
    womenBoxes = [cb["cords"] for cb in womenBoxes]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    loader.mask_predictor.set_image(image_rgb)

    pmasks = []

    for box in womenBoxes:
        masks, scores, logits = loader.mask_predictor.predict(box=box, multimask_output=True)

        highestScoreIndex = np.where(scores == np.amax(scores))[0][0]

        pmasks.append(masks[highestScoreIndex])

    image = Image.open(image_path)
    overlay_image = Image.new("RGBA", image.size, (0, 0, 0, 255))
    overlay_color = (255, 255, 255, 0)

    draw = ImageDraw.Draw(overlay_image)

    segmentation_masks = []

    for mask in pmasks:
        mask = np.array(mask, dtype="int")

        segmentation_mask_image = Image.fromarray(
            (mask * 255).astype("uint8"), mode="L"
        )
        segmentation_masks.append(segmentation_mask_image)

    for segmentation_mask_image in segmentation_masks:
        draw.bitmap((0, 0), segmentation_mask_image, fill=overlay_color)

    mask_image = overlay_image.convert("RGB")
    return mask_image


demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Slider(0, 1, value=0.5, label="clip_threshold"),
        gr.Image(type="filepath"),
    ],
    outputs=["image"],
    allow_flagging="never",
    title="زاهية Segementation",
).queue()

if __name__ == "__main__":

    demo.launch(server_name="localhost", server_port=8413)

