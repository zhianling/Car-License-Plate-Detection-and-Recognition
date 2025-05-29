from ultralytics import YOLO
import torch
import cv2
import warnings
import pandas as pd
import os
import yaml
from torchvision import transforms
import numpy as np  # <--- ADD THIS LINE
# Assuming models.EasyOCR.model and models.EasyOCR.utils are in your project structure:
from models.EasyOCR.utils import AttrDict, AttnLabelConverter, CTCLabelConverter
from models.EasyOCR.model import Model as EasyOCRModel # Renamed to avoid conflict

warnings.filterwarnings("ignore")

# --- Module Specific Configurations ---
# TARGET_CLASS_NAME_YOLO is passed from GUI, e.g., "license_plate"
YOLO_CONFIDENCE_THRESHOLD = 0.25

YOLO_BOX_COLOR = (255, 0, 0)  # Blue for YOLO detected plates
YOLO_TEXT_COLOR = (255, 255, 255)
YOLO_TEXT_BG_COLOR = (0, 0, 0)

OCR_TEXT_COLOR = (50, 200, 255)
OCR_TEXT_BG_COLOR = (0, 0, 0)

# Cache for YOLO model's class names and target class ID
_yolo_class_names_map_cache = None
_yolo_target_cls_id_int_cache = None
_yolo_target_cls_name_cache = None


# === Model Loading Functions ===
def load_yolo_model(model_path, device_pytorch): # device_pytorch is pytorch device, YOLO handles its own
    print(f"YOLO_EasyOCR: Loading YOLOv8 model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"YOLO_EasyOCR: ERROR - YOLOv8 model file not found: {model_path}")
        return None
    try:
        # YOLO internally decides to use CUDA if available based on PyTorch.
        # The 'device' param for YOLO constructor/inference is for explicit override.
        model = YOLO(model_path) # task="detect" is default
        # Perform a dummy inference to populate model.names if not already
        _ = model(np.zeros((224,224,3), dtype=np.uint8), verbose=False)
        print("YOLO_EasyOCR: YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        print(f"YOLO_EasyOCR: Error loading YOLOv8 model: {e}")
        return None

def _get_easyocr_config(file_path): # Make sure this helper function is also present in yolo_easyocr.py
    # ... (implementation of _get_easyocr_config as provided before) ...
    # For example:
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
                except Exception as e_csv: print(f"YOLO_EasyOCR: Warn - CSV err {csv_path}: {e_csv}")
            else: print(f"YOLO_EasyOCR: Warn - CSV missing: {csv_path}")
        opt.character = ''.join(sorted(set(characters)))
    else:
        opt.character = opt.number + opt.symbol + opt.lang_char
    if not opt.character: print("YOLO_EasyOCR: Warning - EasyOCR character set is empty.")
    return opt

def load_easyocr_model(config_path, weight_path, device): # This is the function you want to modify
    print(f"YOLO_EasyOCR: Loading EasyOCR model, Config: {config_path}, Weights: {weight_path}, Device: {device}")
    opt = _get_easyocr_config(config_path)
    opt.input_channel = 1
    if hasattr(opt, 'rgb') and opt.rgb: 
        opt.input_channel = 3
    
    converter = CTCLabelConverter(opt.character) if 'CTC' in opt.Prediction else AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    
    # Ensure EasyOCRModel is correctly imported, e.g., from models.EasyOCR.model import Model as EasyOCRModel
    model = EasyOCRModel(opt) 
    model = torch.nn.DataParallel(model).to(device)
    
    # --- THIS IS THE LINE TO MODIFY ---
    # Original: model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    # Modified:
    model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False), strict=False)
    
    return model.eval(), converter, opt

def prepare_ocr_transform(opt): # Copied from frcnn_easyocr
    print("YOLO_EasyOCR: Preparing EasyOCR transform.")
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((opt.imgH, opt.imgW)),
        transforms.Grayscale(num_output_channels=opt.input_channel),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * opt.input_channel, [0.5] * opt.input_channel)
    ])

# === Frame Processing ===
def run_inference_on_frame(frame_bgr, target_class_name_str, device_pytorch,
                           yolo_model, easyocr_model, converter, ocr_transform_func):
    
    global _yolo_class_names_map_cache, _yolo_target_cls_id_int_cache, _yolo_target_cls_name_cache
    current_yolo_confidence_threshold = YOLO_CONFIDENCE_THRESHOLD 
    # ... (YOLO class map and target ID initialization logic from previous full code) ...
    # ... (Ensure numpy is imported if dummy inference for yolo_model.names is used) ...
    if not hasattr(yolo_model, 'names') or not yolo_model.names:
        try:
            import numpy as np 
            _ = yolo_model(np.zeros((224, 224, 3), dtype=np.uint8), verbose=False)
            if not hasattr(yolo_model, 'names') or not yolo_model.names:
                 print("YOLO_EasyOCR: CRITICAL - YOLO model names not available. Cannot proceed.")
                 return []
        except Exception as e:
            print(f"YOLO_EasyOCR: Error during dummy inference for YOLO names: {e}. Cannot proceed.")
            return []
    if _yolo_class_names_map_cache is None or _yolo_target_cls_name_cache != target_class_name_str.lower() or \
       _yolo_class_names_map_cache != yolo_model.names: 
        _yolo_class_names_map_cache = yolo_model.names
        _yolo_target_cls_name_cache = target_class_name_str.lower()
        _yolo_target_cls_id_int_cache = None 
        for cls_id, name_val in _yolo_class_names_map_cache.items():
            if name_val.lower() == _yolo_target_cls_name_cache:
                _yolo_target_cls_id_int_cache = int(cls_id)
                break
        if _yolo_target_cls_id_int_cache is None:
            print(f"YOLO_EasyOCR: Target class '{target_class_name_str}' not found in YOLO names. Skipping.")
            return []
    if _yolo_target_cls_id_int_cache is None: return []

    yolo_results = yolo_model(frame_bgr, verbose=False, conf=current_yolo_confidence_threshold)
    detections_result = []
    if not yolo_results: return []

    for res in yolo_results:
        for box_data in res.boxes:
            cls_id = int(box_data.cls.item())
            if cls_id == _yolo_target_cls_id_int_cache:
                score = box_data.conf.item()
                xmin, ymin, xmax, ymax = map(int, box_data.xyxy[0].cpu().numpy())
                xmin = max(0, xmin); ymin = max(0, ymin)
                xmax = min(frame_bgr.shape[1] - 1, xmax); ymax = min(frame_bgr.shape[0] - 1, ymax)
                if xmax <= xmin or ymax <= ymin: continue

                plate_roi_bgr = frame_bgr[ymin:ymax, xmin:xmax]
                recognized_text = ""
                pseudo_ocr_confidence = 0.0

                if plate_roi_bgr.size > 0 and easyocr_model and converter and ocr_transform_func:
                    try:
                        ocr_image_tensor = ocr_transform_func(plate_roi_bgr).unsqueeze(0).to(device_pytorch) # Use device_pytorch for EasyOCR
                        with torch.no_grad():
                            ocr_preds_raw = easyocr_model(ocr_image_tensor, '')

                        seq_len_T = ocr_preds_raw.size(1)
                        if seq_len_T > 0:
                            # --- Calculate Pseudo-Confidence using Softmax ---
                            probabilities_batched = torch.softmax(ocr_preds_raw, dim=2)
                            max_probabilities_values, ocr_preds_idx_batched = probabilities_batched.max(2)
                            if max_probabilities_values[0].numel() > 0:
                                pseudo_ocr_confidence = max_probabilities_values[0].mean().item()
                            # --- End Pseudo-Confidence Calculation ---
                            
                            ocr_preds_idx_flattened = ocr_preds_idx_batched.view(-1)
                            
                            batch_size_ocr = ocr_image_tensor.size(0)
                            preds_size_ocr = torch.IntTensor([seq_len_T] * batch_size_ocr).to(device_pytorch)
                            
                            recognized_text_list = converter.decode_greedy(ocr_preds_idx_flattened.data, preds_size_ocr.data)
                            recognized_text = recognized_text_list[0] if recognized_text_list else ""
                            recognized_text = ''.join(filter(str.isalnum, recognized_text)).upper()

                            if not recognized_text:
                                pseudo_ocr_confidence = min(pseudo_ocr_confidence, 0.1)

                    except Exception as e:
                        print(f"YOLO_EasyOCR: Error during OCR: {e}")
                        recognized_text = "OCR_ERR"
                        pseudo_ocr_confidence = 0.0
                
                detections_result.append(((xmin, ymin, xmax, ymax), score, (recognized_text, pseudo_ocr_confidence)))
                
    return detections_result

def display_frame_with_predictions(frame_to_draw, detections):
    global _yolo_class_names_map_cache, _yolo_target_cls_id_int_cache # To get class name for display
    
    # detections is a list of [((box), yolo_score, (ocr_text, ocr_confidence_is_None))]
    for (xmin, ymin, xmax, ymax), yolo_score, (ocr_text, ocr_confidence) in detections:
        
        # --- Draw YOLO Detector Bounding Box and Label ---
        detector_box_color = YOLO_BOX_COLOR
        detector_text_color = YOLO_TEXT_COLOR
        detector_text_bg_color = YOLO_TEXT_BG_COLOR
        
        display_class_name = "plate" # Default
        if _yolo_class_names_map_cache and _yolo_target_cls_id_int_cache is not None:
            display_class_name = _yolo_class_names_map_cache.get(_yolo_target_cls_id_int_cache, "plate")
        label_text_detector = f"{display_class_name}: {yolo_score:.2f}" # Format YOLO score

        cv2.rectangle(frame_to_draw, (xmin, ymin), (xmax, ymax), detector_box_color, 2)
        (text_w_det, text_h_det), base_det = cv2.getTextSize(label_text_detector, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame_to_draw, (xmin, ymin - text_h_det - base_det - 2), 
                      (xmin + text_w_det, ymin - base_det + 2), detector_text_bg_color, -1)
        cv2.putText(frame_to_draw, label_text_detector, (xmin, ymin - base_det -1), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, detector_text_color, 1, cv2.LINE_AA)

        # --- Draw OCR Text (and confidence if available) ---
        if ocr_text: 
            display_ocr_text_final = ocr_text
            if isinstance(ocr_confidence, float): 
                display_ocr_text_final = f"{ocr_text} ({ocr_confidence:.2f})"
            
            ocr_text_y_pos = ymax + 20
            if ocr_text_y_pos + 20 > frame_to_draw.shape[0]: 
                ocr_text_y_pos = ymin - 10
            
            (text_w_ocr, text_h_ocr), base_ocr = cv2.getTextSize(display_ocr_text_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_to_draw, (xmin, ocr_text_y_pos - text_h_ocr - base_ocr), 
                          (xmin + text_w_ocr, ocr_text_y_pos + base_ocr), OCR_TEXT_BG_COLOR, -1)
            cv2.putText(frame_to_draw, display_ocr_text_final, (xmin, ocr_text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCR_TEXT_COLOR, 2)
    return frame_to_draw