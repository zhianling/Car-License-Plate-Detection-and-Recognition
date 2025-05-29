import cv2
import torch
import yaml
import os
import warnings
import numpy as np
import pandas as pd
from torchvision import transforms
# Assuming models.EasyOCR.model and models.EasyOCR.utils are in your project structure:
# your_project_root/models/EasyOCR/model.py
# your_project_root/models/EasyOCR/utils.py
from models.EasyOCR.utils import AttrDict, AttnLabelConverter, CTCLabelConverter
from models.EasyOCR.model import Model as EasyOCRModel # Renamed to avoid conflict if Model is too generic

warnings.filterwarnings("ignore")

# --- Module Specific Configurations ---
FRCNN_CUSTOM_MODEL_CLASS_NAMES = ['__background__', 'carplate']
# TARGET_CLASS_NAME is passed from GUI, e.g., "carplate"
CONFIDENCE_THRESHOLD = 0.5

BOX_COLOR = (0, 0, 255) # Red for car plates (FRCNN)
TEXT_COLOR = (255, 255, 255) # White for FRCNN label
TEXT_BG_COLOR = (0, 0, 0) # Black for FRCNN label background

OCR_TEXT_COLOR = (50, 200, 255)  # Light blue/yellow for EasyOCR text
OCR_TEXT_BG_COLOR = (0,0,0) # Black background for EasyOCR text

# === Model Loading ===
def load_frcnn_model(model_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    print(f"FRCNN_EasyOCR: Loading FRCNN model from {model_path} to {device}")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(FRCNN_CUSTOM_MODEL_CLASS_NAMES))
    
    # MODIFICATION HERE
    checkpoint = torch.load(model_path, map_location=device, weights_only=False) 
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()

def _get_easyocr_config(file_path): # Renamed to avoid direct call if not needed by GUI
    with open(file_path, 'r', encoding="utf8") as stream:
        opt = yaml.safe_load(stream)
    opt = AttrDict(opt)
    if opt.lang_char == 'None':
        characters = ''
        base_train_data_path = opt.get('train_data', 'data/train_data_for_ocr')
        if not os.path.isabs(base_train_data_path) and os.path.dirname(file_path):
            base_train_data_path = os.path.join(os.path.dirname(file_path), base_train_data_path)

        for data_folder in opt.get('select_data', '').split('-'):
            if not data_folder: continue
            csv_path = os.path.join(base_train_data_path, data_folder, 'labels.csv')
            if not os.path.exists(csv_path): csv_path = os.path.join(base_train_data_path, data_folder, 'label.csv')
            
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, sep='^([^,]+),', engine='python', usecols=['filename', 'words'], keep_default_na=False)
                    all_char = ''.join(df['words'].astype(str))
                    characters += ''.join(set(all_char))
                except Exception as e_csv: print(f"FRCNN_EasyOCR: Warn - CSV err {csv_path}: {e_csv}")
            else: print(f"FRCNN_EasyOCR: Warn - CSV missing: {csv_path}")
        opt.character = ''.join(sorted(set(characters)))
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    if not opt.character: print("FRCNN_EasyOCR: Warning - EasyOCR character set is empty.")
    return opt

def load_easyocr_model(config_path, weight_path, device):
    print(f"FRCNN_EasyOCR: Loading EasyOCR model, Config: {config_path}, Weights: {weight_path}, Device: {device}")
    opt = _get_easyocr_config(config_path)
    opt.input_channel = 1 
    if hasattr(opt, 'rgb') and opt.rgb: 
        opt.input_channel = 3

    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.Prediction else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    model = EasyOCRModel(opt) 
    model = torch.nn.DataParallel(model).to(device) 
    
    # MODIFICATION HERE
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False), strict=False) 
    
    return model.eval(), converter, opt

def prepare_ocr_transform(opt):
    print("FRCNN_EasyOCR: Preparing EasyOCR transform.")
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.imgH, opt.imgW)),
        transforms.Grayscale(num_output_channels=opt.input_channel),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * opt.input_channel, [0.5] * opt.input_channel) # Use opt.input_channel for normalize
    ])

# === Frame Processing ===
def run_inference_on_frame(frame_bgr, target_class_name_str, device,
                           frcnn_model, easyocr_model, converter, ocr_transform_func):
    
    # ... (FRCNN detection logic from previous full code to get xmin, ymin, xmax, ymax, score, target_class_id) ...
    # Example snippet assuming FRCNN part is already there:
    current_frcnn_class_names = FRCNN_CUSTOM_MODEL_CLASS_NAMES
    current_confidence_threshold = CONFIDENCE_THRESHOLD
    try:
        target_class_id = current_frcnn_class_names.index(target_class_name_str.lower())
    except ValueError:
        print(f"FRCNN_EasyOCR: Target class '{target_class_name_str}' not in {current_frcnn_class_names}. Skipping.")
        return []
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_tensor_chw = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().to(device) / 255.0
    with torch.no_grad():
        frcnn_outputs = frcnn_model([img_tensor_chw])[0]
    detections_result = []
    pred_boxes = frcnn_outputs['boxes'].cpu().numpy()
    pred_labels = frcnn_outputs['labels'].cpu().numpy()
    pred_scores = frcnn_outputs['scores'].cpu().numpy()

    for i in range(len(pred_scores)):
        score = pred_scores[i]
        label_id = pred_labels[i]
        if label_id == target_class_id and score >= current_confidence_threshold:
            box = pred_boxes[i]
            xmin, ymin, xmax, ymax = map(int, box)
            xmin = max(0, xmin); ymin = max(0, ymin)
            xmax = min(frame_bgr.shape[1] - 1, xmax); ymax = min(frame_bgr.shape[0] - 1, ymax)
            if xmax <= xmin or ymax <= ymin: continue

            plate_roi_bgr = frame_bgr[ymin:ymax, xmin:xmax]
            recognized_text = ""
            pseudo_ocr_confidence = 0.0 

            if plate_roi_bgr.size > 0 and easyocr_model and converter and ocr_transform_func:
                try:
                    ocr_image_tensor = ocr_transform_func(plate_roi_bgr).unsqueeze(0).to(device)
                    with torch.no_grad():
                        ocr_preds_raw = easyocr_model(ocr_image_tensor, '') 

                    seq_len_T = ocr_preds_raw.size(1)
                    if seq_len_T > 0:
                        # --- Calculate Pseudo-Confidence using Softmax ---
                        # 1. Apply Softmax to get probabilities
                        probabilities_batched = torch.softmax(ocr_preds_raw, dim=2) # dim=2 is the character class dimension
                        
                        # 2. Get the probabilities of the chosen characters at each time step
                        max_probabilities_values, ocr_preds_idx_batched = probabilities_batched.max(2)
                        # max_probabilities_values has shape [batch_size, seq_len_T]
                        
                        # 3. Average these max probabilities for the sequence
                        # (This is still a simplification as it doesn't perfectly align with CTC decoded path)
                        if max_probabilities_values[0].numel() > 0: # For the first (and only) batch item
                            pseudo_ocr_confidence = max_probabilities_values[0].mean().item()
                        # --- End Pseudo-Confidence Calculation ---
                        
                        ocr_preds_idx_flattened = ocr_preds_idx_batched.view(-1) # Used for decoding
                        
                        # --- Decode Text ---
                        batch_size_ocr = ocr_image_tensor.size(0)
                        preds_size_ocr = torch.IntTensor([seq_len_T] * batch_size_ocr).to(device)
                        
                        recognized_text_list = converter.decode_greedy(ocr_preds_idx_flattened.data, preds_size_ocr.data)
                        recognized_text = recognized_text_list[0] if recognized_text_list else ""
                        recognized_text = ''.join(filter(str.isalnum, recognized_text)).upper()

                        if not recognized_text: # If cleaning results in empty text
                            pseudo_ocr_confidence = min(pseudo_ocr_confidence, 0.1) # Reduce confidence

                except Exception as e:
                    print(f"FRCNN_EasyOCR: Error during OCR: {e}")
                    recognized_text = "OCR_ERR"
                    pseudo_ocr_confidence = 0.0
            
            detections_result.append(((xmin, ymin, xmax, ymax), score, (recognized_text, pseudo_ocr_confidence)))
            
    return detections_result

def display_frame_with_predictions(frame_to_draw, detections):
    # detections is a list of [((box), frcnn_score, (ocr_text, ocr_confidence_is_None))]
    for (xmin, ymin, xmax, ymax), frcnn_score, (ocr_text, ocr_confidence) in detections:
        
        # --- Draw FRCNN Detector Bounding Box and Label ---
        detector_box_color = BOX_COLOR
        detector_text_color = TEXT_COLOR
        detector_text_bg_color = TEXT_BG_COLOR
        display_class_name = FRCNN_CUSTOM_MODEL_CLASS_NAMES[1] if len(FRCNN_CUSTOM_MODEL_CLASS_NAMES) > 1 else "plate"
        label_text_detector = f"{display_class_name}: {frcnn_score:.2f}" # Format FRCNN score

        cv2.rectangle(frame_to_draw, (xmin, ymin), (xmax, ymax), detector_box_color, 2)
        (text_w_det, text_h_det), base_det = cv2.getTextSize(label_text_detector, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_to_draw, (xmin, ymin - text_h_det - base_det - 2), 
                      (xmin + text_w_det, ymin - base_det + 2), detector_text_bg_color, -1)
        cv2.putText(frame_to_draw, label_text_detector, (xmin, ymin - base_det -1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, detector_text_color, 1, cv2.LINE_AA)

        # --- Draw OCR Text (and confidence if available) ---
        if ocr_text: # Only proceed if there's some OCR text (EasyOCR returns text or empty)
            display_ocr_text_final = ocr_text
            # For EasyOCR, ocr_confidence is None from run_inference_on_frame,
            # so this 'if' block for confidence won't be entered.
            # If you later modify EasyOCR to return a float confidence, this will display it.
            if isinstance(ocr_confidence, float): 
                display_ocr_text_final = f"{ocr_text} ({ocr_confidence:.2f})" 
            
            ocr_text_y_pos = ymax + 20
            # Adjust y_pos if it goes off-screen
            if ocr_text_y_pos + 20 > frame_to_draw.shape[0]: # +20 for approximate text height
                ocr_text_y_pos = ymin - 10 
            
            (text_w_ocr, text_h_ocr), base_ocr = cv2.getTextSize(display_ocr_text_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_to_draw, (xmin, ocr_text_y_pos - text_h_ocr - base_ocr), 
                          (xmin + text_w_ocr, ocr_text_y_pos + base_ocr), OCR_TEXT_BG_COLOR, -1)
            cv2.putText(frame_to_draw, display_ocr_text_final, (xmin, ocr_text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCR_TEXT_COLOR, 2)
    return frame_to_draw