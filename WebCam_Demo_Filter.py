import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    for box, label in zip(boxes, labels):
        box = box * torch.tensor([W, H, W, H])
        box[:2] -= box[2:] / 2  # Convert center to top-left
        box[2:] += box[:2]      # Convert width/height to bottom-right
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        font = ImageFont.load_default()
        draw.text((x0, y0), str(label), fill="white", font=font)
        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=3)

    return image_pil, mask


def preprocess_image(frame):
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(f"Model loaded with result: {load_res}")
    _ = model.eval()
    return model


def get_grounding_output(
    model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None
):
    assert (
        text_threshold is not None or token_spans is not None
    ), "text_threshold and token_spans should not be None at the same time!"
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, num_classes)
    boxes = outputs["pred_boxes"][0]              # (num_queries, 4)

    if token_spans is None:
        logits_filt = logits.cpu()
        boxes_filt = boxes.cpu()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f" ({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # Handling token spans if provided (not used in this example)
        pass

    return boxes_filt, pred_phrases


if __name__ == "__main__":
    config_file = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
    checkpoint_path = "weights/groundingdino_swint_ogc.pth"
    text_prompt = "the fan.the light.the speaker"
    box_threshold = 0.3
    text_threshold = 0.25
    cpu_only = False

    model = load_model(config_file, checkpoint_path, cpu_only=cpu_only)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Unable to capture frame.")
            break

        image_pil, image = preprocess_image(frame)

        boxes_filt, pred_phrases = get_grounding_output(
            model, image, text_prompt, box_threshold, text_threshold, cpu_only=cpu_only
        )

        size = image_pil.size  # (W, H)
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # [H, W]
            "labels": pred_phrases,
        }

        # Initialize lists for filtered detections
        filtered_boxes = []
        filtered_labels = []

        H, W = pred_dict['size']

        # Loop over detections
        for box, label in zip(pred_dict['boxes'], pred_dict['labels']):
            # Convert box coordinates from normalized center to pixel coordinates
            box_pixel = box * torch.tensor([W, H, W, H])
            cx, cy, w, h = box_pixel
            x0 = cx - w / 2
            y0 = cy - h / 2
            x1 = cx + w / 2
            y1 = cy + h / 2
            width = x1 - x0
            height = y1 - y0

            # Calculate aspect ratios
            aspect_ratio_width_height = width / height
            aspect_ratio_height_width = height / width

            # Filtering based on class
            label_lower = label.lower()
            if 'light' in label_lower:
                if aspect_ratio_height_width >= 1.1:  # Height is at least 1.1 times the width
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            elif 'fan' in label_lower:
                if aspect_ratio_height_width >= 1.2:  # Height is at least 1.2 times the width
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            elif 'speaker' in label_lower:
                if aspect_ratio_width_height >= 1.1:  # Width is at least 1.1 times the height
                    filtered_boxes.append(box)
                    filtered_labels.append(label)
            else:
                # Optionally include other classes or skip
                pass

        # Update pred_dict with filtered detections
        if filtered_boxes:
            pred_dict['boxes'] = torch.stack(filtered_boxes)
            pred_dict['labels'] = filtered_labels
        else:
            pred_dict['boxes'] = torch.empty((0, 4))
            pred_dict['labels'] = []

        # Plot the filtered boxes on the image
        image_with_box = plot_boxes_to_image(image_pil, pred_dict)[0]

        image_with_box_cv2 = cv2.cvtColor(np.array(image_with_box), cv2.COLOR_RGB2BGR)
        cv2.imshow('Webcam Demo', image_with_box_cv2)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
