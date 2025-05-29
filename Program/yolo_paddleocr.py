from ultralytics import YOLO
import torch # Still needed if other parts of the GUI use it for device detection
import paddle
from paddleocr import PaddleOCR as PaddleOCR_lib
import cv2
import warnings
import os
import numpy as np # For dummy YOLO inference

warnings.filterwarnings("ignore")

# --- Module Specific Configurations ---
# TARGET_CLASS_NAME_YOLO is passed from GUI, e.g., "license_plate"
YOLO_CONFIDENCE_THRESHOLD = 0.25

YOLO_BOX_COLOR = (255, 0, 0)  # Blue for YOLO detected plates
YOLO_TEXT_COLOR = (255, 255, 255)
YOLO_TEXT_BG_COLOR = (0, 0, 0)

OCR_TEXT_COLOR = (0, 255, 0) # Green for PaddleOCR text
OCR_TEXT_BG_COLOR = (0,0,0)

# Cache for YOLO model's class names and target class ID
_yolo_class_names_map_cache = None
_yolo_target_cls_id_int_cache = None
_yolo_target_cls_name_cache = None


# === Model Loading Functions ===
def load_yolo_model(model_path, device_pytorch_ignored): # device_pytorch is not directly used by YOLO loader here
    print(f"YOLO_PaddleOCR: Loading YOLOv8 model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"YOLO_PaddleOCR: ERROR - YOLOv8 model file not found: {model_path}")
        return None
    try:
        model = YOLO(model_path)
        _ = model(np.zeros((224,224,3), dtype=np.uint8), verbose=False) # Dummy inference for names
        print("YOLO_PaddleOCR: YOLOv8 model loaded successfully.")
        return model
    except Exception as e:
        print(f"YOLO_PaddleOCR: Error loading YOLOv8 model: {e}")
        return None

def load_paddleocr_model(lang='en', use_angle_cls=True, use_gpu_paddle=True): # use_gpu_paddle comes from GUI based on self.device
    print(f"Module_PaddleOCR: Initializing PaddleOCR with lang='{lang}', use_angle_cls={use_angle_cls}, use_gpu_flag_from_gui={use_gpu_paddle}.")
    
    actual_use_gpu_paddle = False # What PaddleOCR will actually use
    paddle_device_str = 'cpu'

    if use_gpu_paddle and paddle.is_compiled_with_cuda():
        num_gpus_paddle = paddle.device.cuda.device_count()
        if num_gpus_paddle > 0:
            # Try to find an NVIDIA GPU if multiple GPUs are present
            # This is a heuristic. A truly robust way would involve checking vendor IDs.
            selected_gpu_id = 0 # Default to the first GPU Paddle sees
            best_gpu_name = ""

            if num_gpus_paddle > 0: # If there's at least one CUDA device for Paddle
                for i in range(num_gpus_paddle):
                    props = paddle.device.cuda.get_device_properties(i)
                    print(f"PaddleOCR: Found GPU {i}: {props.name}")
                    if "nvidia" in props.name.lower(): # Prefer NVIDIA
                        selected_gpu_id = i
                        best_gpu_name = props.name
                        break # Found an NVIDIA, use it
                    if not best_gpu_name: # If no NVIDIA found yet, take the first one
                        best_gpu_name = props.name
                        selected_gpu_id = i # Will be 0 if only one non-NVIDIA CUDA GPU

                paddle_device_str = f'gpu:{selected_gpu_id}'
                actual_use_gpu_paddle = True
                print(f"PaddleOCR: Selected GPU {selected_gpu_id}: {best_gpu_name} for Paddle. Setting device to {paddle_device_str}")
            else: # No CUDA GPUs found by Paddle
                print("PaddleOCR: Paddle compiled with CUDA, but no CUDA devices found by paddle.device.cuda.device_count(). Using CPU.")
                paddle_device_str = 'cpu'
                actual_use_gpu_paddle = False
        else: # Not compiled with CUDA or use_gpu_paddle was false
            print("PaddleOCR: Not using GPU (either flag was false or Paddle not compiled with CUDA). Using CPU.")
            paddle_device_str = 'cpu'
            actual_use_gpu_paddle = False
    else: # Main use_gpu_paddle flag from GUI was false
        print("PaddleOCR: GPU usage not requested by GUI. Using CPU.")
        paddle_device_str = 'cpu'
        actual_use_gpu_paddle = False

    try:
        paddle.set_device(paddle_device_str)
        print(f"PaddleOCR: paddle.set_device('{paddle_device_str}') called.")
        
        ocr_instance = PaddleOCR_lib(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=actual_use_gpu_paddle, # This flag is crucial for PaddleOCR
            show_log=False
        )
        print(f"PaddleOCR: Instance created. Effective GPU use: {actual_use_gpu_paddle}")
        return ocr_instance
    except Exception as e:
        print(f"Module_PaddleOCR: Error initializing PaddleOCR: {e}")
        print("Module_PaddleOCR: Falling back to CPU for PaddleOCR if possible or failing.")
        try: # Attempt CPU fallback on error
            paddle.set_device('cpu')
            ocr_instance = PaddleOCR_lib(
                use_angle_cls=use_angle_cls, lang=lang, use_gpu=False, show_log=False
            )
            print("Module_PaddleOCR: Successfully fell back to CPU for PaddleOCR.")
            return ocr_instance
        except Exception as e2:
            print(f"Module_PaddleOCR: CPU fallback for PaddleOCR also failed: {e2}")
            return None

# === Frame Processing ===
def run_inference_on_frame(frame_bgr, target_class_name_str, device_pytorch_ignored, # device_pytorch is not directly used by YOLO inference here
                           yolo_model, paddle_ocr_instance,
                           _converter_ignored=None, _ocr_transform_ignored=None): # Ignored for API consistency
    
    global _yolo_class_names_map_cache, _yolo_target_cls_id_int_cache, _yolo_target_cls_name_cache

    # These should be defined at the module level or passed if they can change
    current_yolo_confidence_threshold = YOLO_CONFIDENCE_THRESHOLD # Example: 0.25

    # Initialize/Update YOLO class map and target ID if model or target name changed
    # This logic ensures we only try to find the target class ID once per model/target_name
    if not hasattr(yolo_model, 'names') or not yolo_model.names:
        # Perform a dummy inference if names are not populated (might be needed for some YOLO versions/models)
        # This assumes yolo_model is an Ultralytics YOLO object
        try:
            import numpy as np # Ensure numpy is imported
            _ = yolo_model(np.zeros((224, 224, 3), dtype=np.uint8), verbose=False)
            if not hasattr(yolo_model, 'names') or not yolo_model.names:
                 print("YOLO_PaddleOCR: CRITICAL - YOLO model names not available even after dummy inference. Cannot proceed.")
                 return []
        except Exception as e:
            print(f"YOLO_PaddleOCR: Error during dummy inference for YOLO names: {e}. Cannot proceed.")
            return []


    if _yolo_class_names_map_cache is None or _yolo_target_cls_name_cache != target_class_name_str.lower() or \
       _yolo_class_names_map_cache != yolo_model.names: # Check if model itself changed
        _yolo_class_names_map_cache = yolo_model.names
        _yolo_target_cls_name_cache = target_class_name_str.lower()
        _yolo_target_cls_id_int_cache = None # Reset
        for cls_id, name_val in _yolo_class_names_map_cache.items():
            if name_val.lower() == _yolo_target_cls_name_cache:
                _yolo_target_cls_id_int_cache = int(cls_id)
                print(f"YOLO_PaddleOCR: Target class '{target_class_name_str}' (ID: {_yolo_target_cls_id_int_cache}) initialized from model names: {_yolo_class_names_map_cache}")
                break
        if _yolo_target_cls_id_int_cache is None:
            print(f"YOLO_PaddleOCR: Target class '{target_class_name_str}' not found in YOLO model names: {_yolo_class_names_map_cache}. Skipping YOLO detections.")
            return [] # Cannot proceed if target class ID is not found
    
    if _yolo_target_cls_id_int_cache is None: # Double check if target ID is still None
        # This case should ideally be caught above, but as a safeguard:
        print(f"YOLO_PaddleOCR: Target class ID for '{target_class_name_str}' not resolved. Skipping.")
        return []

    # YOLO inference. Ultralytics YOLO handles its own device (CPU/GPU) based on availability
    # The `device` argument in yolo_model() call can explicitly set it, e.g., device='cpu' or device='0' for GPU 0
    # For simplicity, we let YOLO decide or use its default (often GPU if available)
    yolo_results = yolo_model(frame_bgr, verbose=False, conf=current_yolo_confidence_threshold)
    detections_result = []

    # yolo_results is a list of Results objects. For a single image/frame, it will have one element.
    if not yolo_results:
        return []

    for res in yolo_results: # Iterate through results for each image (though it's one frame)
        for box_data in res.boxes: # Access the Boxes object
            cls_id = int(box_data.cls.item()) # Get class ID
            
            if cls_id == _yolo_target_cls_id_int_cache:
                score = box_data.conf.item() # Get confidence score
                # Get bounding box coordinates (xyxy format)
                xmin, ymin, xmax, ymax = map(int, box_data.xyxy[0].cpu().numpy()) 
                
                # Clamp coordinates
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(frame_bgr.shape[1] - 1, xmax)
                ymax = min(frame_bgr.shape[0] - 1, ymax)
                
                if xmax <= xmin or ymax <= ymin: # Invalid box
                    continue

                plate_roi_bgr = frame_bgr[ymin:ymax, xmin:xmax]
                
                # --- Revised PaddleOCR text and confidence extraction ---
                ocr_text_content = ""
                ocr_confidence_score = 0.0
                
                if plate_roi_bgr.size > 0 and paddle_ocr_instance:
                    try:
                        ocr_result = paddle_ocr_instance.ocr(plate_roi_bgr, det=False, rec=True, cls=paddle_ocr_instance.use_angle_cls)
                        if ocr_result and ocr_result[0]: 
                            lines_info = ocr_result[0]
                            if lines_info : 
                                recognized_text_tuple = lines_info[0]
                                text_content_raw = recognized_text_tuple[0]
                                ocr_confidence_score = recognized_text_tuple[1]
                                ocr_text_content = ''.join(filter(str.isalnum, text_content_raw)).upper()
                    except Exception as e:
                        print(f"YOLO_PaddleOCR: Error during PaddleOCR: {e}")
                        ocr_text_content = "OCR_ERR"
                        ocr_confidence_score = 0.0
                
                detections_result.append(((xmin, ymin, xmax, ymax), score, (ocr_text_content, ocr_confidence_score)))
                # --- End of revised section ---
                
    return detections_result

# In yolo_paddleocr.py

def display_frame_with_predictions(frame_to_draw, detections):
    global _yolo_class_names_map_cache, _yolo_target_cls_id_int_cache # To get class name for display

    # detections is a list of [((box), yolo_score, (ocr_text, ocr_confidence_score))]
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
        if ocr_text or ocr_text == "OCR_ERR": 
            display_ocr_text_final = ocr_text
            if ocr_text != "OCR_ERR" and isinstance(ocr_confidence, float): 
                 display_ocr_text_final = f"{ocr_text} ({ocr_confidence:.2f})" # Format PaddleOCR confidence
            
            ocr_text_y_pos = ymax + 20
            if ocr_text_y_pos + 20 > frame_to_draw.shape[0]: 
                ocr_text_y_pos = ymin - 10
            
            (text_w_ocr, text_h_ocr), base_ocr = cv2.getTextSize(display_ocr_text_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_to_draw, (xmin, ocr_text_y_pos - text_h_ocr - base_ocr), 
                          (xmin + text_w_ocr, ocr_text_y_pos + base_ocr), OCR_TEXT_BG_COLOR, -1)
            cv2.putText(frame_to_draw, display_ocr_text_final, (xmin, ocr_text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCR_TEXT_COLOR, 2)
    return frame_to_draw