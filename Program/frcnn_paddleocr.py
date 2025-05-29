import torch
import torchvision # Keep for FRCNN model definition
import paddle
from paddleocr import PaddleOCR as PaddleOCR_lib # Alias to avoid conflict with any local PaddleOCR class
import cv2
import warnings
import os

warnings.filterwarnings("ignore")

# --- Module Specific Configurations ---
FRCNN_CUSTOM_MODEL_CLASS_NAMES = ['__background__', 'carplate']
# TARGET_CLASS_NAME is passed from GUI, e.g., "carplate"
CONFIDENCE_THRESHOLD = 0.5

BOX_COLOR = (0, 0, 255)  # Red for FRCNN car plates
TEXT_COLOR = (255, 255, 255)  # White for FRCNN label
TEXT_BG_COLOR = (0, 0, 0)  # Black for FRCNN label background

OCR_TEXT_COLOR = (0, 255, 0)  # Green for PaddleOCR text
OCR_TEXT_BG_COLOR = (0, 0, 0) # Black background for PaddleOCR text

# === Model Loading Functions ===
def load_frcnn_model(model_path, device):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    print(f"FRCNN_PaddleOCR: Loading FRCNN model from {model_path} to {device}")
    model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(FRCNN_CUSTOM_MODEL_CLASS_NAMES))
    # model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=len(FRCNN_CUSTOM_MODEL_CLASS_NAMES)) # Older
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device).eval()

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
def run_inference_on_frame(frame_bgr, target_class_name_str, device, # PyTorch device for FRCNN
                           frcnn_model, paddle_ocr_instance,
                           _converter_ignored=None, _ocr_transform_ignored=None): # Ignored args for API consistency
    
    current_frcnn_class_names = FRCNN_CUSTOM_MODEL_CLASS_NAMES # Example: ['__background__', 'carplate']
    current_confidence_threshold = CONFIDENCE_THRESHOLD   # Example: 0.5

    try:
        target_class_id = current_frcnn_class_names.index(target_class_name_str.lower())
    except ValueError:
        print(f"FRCNN_PaddleOCR: Target class '{target_class_name_str}' not in {current_frcnn_class_names}. Skipping FRCNN detections.")
        return []

    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Standard torchvision transform for FasterRCNN
    # Ensure img_tensor_chw is on the correct device for frcnn_model
    img_tensor_chw = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float().to(device) / 255.0
    
    with torch.no_grad():
        frcnn_outputs = frcnn_model([img_tensor_chw])[0] # Get predictions for the first (and only) image

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
            
            # Clamp coordinates to be within frame boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(frame_bgr.shape[1] - 1, xmax) 
            ymax = min(frame_bgr.shape[0] - 1, ymax) 
            
            if xmax <= xmin or ymax <= ymin: # Check for valid box dimensions
                continue

            plate_roi_bgr = frame_bgr[ymin:ymax, xmin:xmax]
            
            # --- Revised PaddleOCR text and confidence extraction ---
            ocr_text_content = ""
            ocr_confidence_score = 0.0 # Default to 0.0 for confidence
            
            if plate_roi_bgr.size > 0 and paddle_ocr_instance: # Ensure ROI is not empty
                try:
                    # PaddleOCR expects BGR image (numpy array)
                    ocr_result = paddle_ocr_instance.ocr(plate_roi_bgr, det=False, rec=True, cls=paddle_ocr_instance.use_angle_cls)
                    
                    if ocr_result and ocr_result[0]: 
                        lines_info = ocr_result[0] # This is the list of recognized lines for the ROI
                        if lines_info : # If lines_info is not None or empty
                            # Assuming the first recognized line is the primary one for a plate
                            recognized_text_tuple = lines_info[0] 
                            text_content_raw = recognized_text_tuple[0]
                            ocr_confidence_score = recognized_text_tuple[1] # This is the confidence
                            
                            # Clean the recognized text
                            ocr_text_content = ''.join(filter(str.isalnum, text_content_raw)).upper()
                            
                except Exception as e:
                    print(f"FRCNN_PaddleOCR: Error during PaddleOCR for a plate ROI: {e}")
                    ocr_text_content = "OCR_ERR" # Indicate error
                    ocr_confidence_score = 0.0 # Set confidence to 0 on error
            
            detections_result.append(((xmin, ymin, xmax, ymax), score, (ocr_text_content, ocr_confidence_score)))
            
    return detections_result

def display_frame_with_predictions(frame_to_draw, detections):
    # detections is a list of [((box), frcnn_score, (ocr_text, ocr_confidence_score))]
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
        # For PaddleOCR, ocr_text can be "OCR_ERR" or actual text. ocr_confidence is a float.
        if ocr_text or ocr_text == "OCR_ERR": # Proceed if any ocr_text is present
            display_ocr_text_final = ocr_text
            # Display confidence if text is not an error and confidence is a valid number
            if ocr_text != "OCR_ERR" and isinstance(ocr_confidence, float): 
                 display_ocr_text_final = f"{ocr_text} ({ocr_confidence:.2f})" # Format PaddleOCR confidence
            
            ocr_text_y_pos = ymax + 20
            if ocr_text_y_pos + 20 > frame_to_draw.shape[0]: # Account for text height
                ocr_text_y_pos = ymin - 10 
            
            (text_w_ocr, text_h_ocr), base_ocr = cv2.getTextSize(display_ocr_text_final, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame_to_draw, (xmin, ocr_text_y_pos - text_h_ocr - base_ocr), 
                          (xmin + text_w_ocr, ocr_text_y_pos + base_ocr), OCR_TEXT_BG_COLOR, -1)
            cv2.putText(frame_to_draw, display_ocr_text_final, (xmin, ocr_text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, OCR_TEXT_COLOR, 2)
    return frame_to_draw