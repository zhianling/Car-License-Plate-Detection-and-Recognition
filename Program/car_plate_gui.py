import os
import certifi

# Apply SSL fix at the very beginning
os.environ['SSL_CERT_FILE'] = certifi.where()

import torch
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import importlib
import sys # For path manipulation if needed (not strictly required with current structure)

class CarPlateApp:
    def __init__(self, root_window):
        self.root = root_window
        self.root.title("Car Plate Detection GUI")

        window_width = 450
        window_height = 350
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # --- PyTorch Device Setup ---
        # With CUDA_VISIBLE_DEVICES set, torch.cuda.is_available() will check
        # the visibility of the specified GPU(s). If '1' is your NVIDIA, it becomes cuda:0.
        if torch.cuda.is_available(): # This will now see all available CUDA GPUs
            self.device = torch.device('cuda:0') # Default to the first CUDA GPU PyTorch sees
            print(f"GUI: PyTorch is using device: {self.device}")
            try:
                print(f"GUI: PyTorch CUDA device name: {torch.cuda.get_device_name(0)}")
                if torch.cuda.device_count() > 1:
                    print(f"GUI: PyTorch found multiple CUDA GPUs: {torch.cuda.device_count()}. Using cuda:0 by default.")
            except Exception as e:
                print(f"GUI: Could not get PyTorch CUDA device name: {e}")
        else:
            self.device = torch.device('cpu')
            print(f"GUI: PyTorch is using device: {self.device} (CUDA not available)")

        # --- File Selection ---
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=(15, 5))

        self.open_btn = tk.Button(file_frame, text="Open Image/Video", command=self.open_file, width=20)
        self.open_btn.pack(pady=(0, 5))

        self.file_label = tk.Label(file_frame, text="No file selected", width=30, anchor='center', justify='center')
        self.file_label.pack()

        # --- Model Choice ---
        self.model_choice = tk.StringVar()
        self.model_choice.set("FasterRCNN + EasyOCR") # Default choice

        model_label_frame = tk.Frame(self.root)
        model_label_frame.pack(pady=(10,0))
        tk.Label(model_label_frame, text="Choose Model Combination:").pack()

        radio_frame = tk.Frame(self.root)
        radio_frame.pack(pady=5)

        options = [
            "FasterRCNN + EasyOCR", "FasterRCNN + PaddleOCR",
            "YOLOv8 + EasyOCR", "YOLOv8 + PaddleOCR"
        ]
        for opt in options:
            rb = tk.Radiobutton(radio_frame, text=opt, variable=self.model_choice, value=opt,
                                command=self.on_model_choice_change)
            rb.pack(anchor='w', padx=20) # anchor west for alignment

        self.submit_btn = tk.Button(self.root, text="Submit & Process", command=self.submit_choice, width=20)
        self.submit_btn.pack(pady=(10, 15))

        # Placeholder for image/video status
        self.status_label = tk.Label(self.root, text="Initializing...")
        self.status_label.pack(pady=5)

        self.loaded_path = None
        self.loaded_frame_for_image = None # Store the loaded image frame
        self.is_video = False

        self.detector_module = None
        self.model_objects = {} # To store loaded model instances
        self.active_video_window = None # To keep track of the video display window

        # Preload models for the default choice
        # This will run after the __init__ method has set up the basic GUI elements.
        self.root.after(100, self.on_model_choice_change) # Use after to allow GUI to draw first


    def on_model_choice_change(self):
        """Loads models when the radio button selection changes."""
        self.status_label.config(text=f"Loading models for: {self.model_choice.get()}...")
        self.root.update_idletasks() # Force GUI update
        try:
            choice = self.model_choice.get()
            self.detector_module = self.get_detector_module(choice)
            self.model_objects = self.load_models_for_module(choice)
            self.status_label.config(text=f"Models loaded for: {self.model_choice.get()}. Ready.")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Failed to load models for {self.model_choice.get()}:\n{e}")
            self.status_label.config(text=f"Error loading models for {self.model_choice.get()}.")
            print(f"Error loading models: {e}")


    def get_detector_module(self, choice_str):
        mapping = {
            "FasterRCNN + EasyOCR": "frcnn_easyocr",
            "FasterRCNN + PaddleOCR": "frcnn_paddleocr",
            "YOLOv8 + EasyOCR": "yolo_easyocr",
            "YOLOv8 + PaddleOCR": "yolo_paddleocr"
        }
        module_name = mapping.get(choice_str)
        if not module_name:
            raise ValueError(f"Unsupported model choice string: {choice_str}")
        
        # Invalidate Python's cache for the module to ensure it's reloaded if changed
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        return importlib.import_module(module_name)

    def load_models_for_module(self, choice_str):
        m = self.detector_module
        objs = {}
        print(f"\nLoading models for: {choice_str} using module: {m.__name__}")

        # Determine detector and OCR types from choice_str
        is_frcnn = "FasterRCNN" in choice_str
        is_yolo = "YOLOv8" in choice_str
        is_easyocr = "EasyOCR" in choice_str
        is_paddleocr = "PaddleOCR" in choice_str

        if is_frcnn and hasattr(m, 'load_frcnn_model'):
            print("Loading Faster R-CNN model...")
            objs['detector'] = m.load_frcnn_model(
                'models/full_fasterrcnn_best.pth',
                self.device
            )
        elif is_yolo and hasattr(m, 'load_yolo_model'):
            print("Loading YOLOv8 model...")
            objs['detector'] = m.load_yolo_model(
                'models/best_yolov8.pt',
                self.device # YOLO model loading in ultralytics handles device internally
            )

        if is_easyocr and hasattr(m, 'load_easyocr_model'):
            print("Loading EasyOCR model...")
            # load_easyocr_model should return (model, converter, opt)
            ocr_model, converter, opt = m.load_easyocr_model(
                'models/EasyOCR/opt.txt',
                'models/best_easyocr_full.pth',
                self.device
            )
            objs['ocr'] = ocr_model
            objs['converter'] = converter
            objs['opt'] = opt
            if hasattr(m, 'prepare_ocr_transform') and 'opt' in objs:
                print("Preparing EasyOCR transform...")
                objs['ocr_transform'] = m.prepare_ocr_transform(objs['opt'])

        elif is_paddleocr and hasattr(m, 'load_paddleocr_model'):
            print("Loading PaddleOCR model...")
            use_gpu_paddle = True if self.device.type == 'cuda' else False
            paddle_ocr_instance = m.load_paddleocr_model(
                lang='en',
                use_angle_cls=True,
                use_gpu_paddle=use_gpu_paddle
            )
            if paddle_ocr_instance is not None:
                objs['ocr'] = paddle_ocr_instance
            else:
                print("Warning: PaddleOCR model failed to load.")
                raise RuntimeError("PaddleOCR model failed to load.")


        print(f"Loaded model objects: {list(objs.keys())}")
        if 'detector' not in objs or 'ocr' not in objs:
            missing = []
            if 'detector' not in objs: missing.append("detector")
            if 'ocr' not in objs: missing.append("OCR")
            raise RuntimeError(f"Failed to load required models: {', '.join(missing)} missing for {choice_str}")
        return objs

    def process_frame(self, frame):
        m = self.detector_module
        objs = self.model_objects
        
        target_class_name_for_module = "carplate" # General hint
        if "YOLOv8" in self.model_choice.get():
             target_class_name_for_module = "license_plate"

        detections = m.run_inference_on_frame(
            frame,
            target_class_name_for_module, # Generic target name, module should adapt
            self.device,
            objs['detector'],           # Detector model (FRCNN or YOLO)
            objs['ocr'],                # OCR model (EasyOCR or PaddleOCR)
            objs.get('converter'),      # Specific to EasyOCR
            objs.get('ocr_transform')   # Specific to EasyOCR
        )
        # The display function in each module will use its own specific class names for drawing boxes
        frame_out = m.display_frame_with_predictions(frame.copy(), detections)
        return frame_out


    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=(
                # Option 1: Add a combined "Media files" type first
                ("Media files", "*.jpg *.jpeg *.png *.bmp *.tiff *.mp4 *.avi *.mov *.mkv"),
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")

                # Option 2: Or, just reorder to put "All files" first if you prefer that as default
                # ("All files", "*.*"),
                # ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                # ("Video files", "*.mp4 *.avi *.mov *.mkv"),
            )
        )
        if not path:
            self.file_label.config(text="No file selected") # Update label if dialog cancelled
            return

        self.loaded_path = path
        self.file_label.config(text=os.path.basename(path))
        self.is_video = path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))

        if not self.is_video:
            self.loaded_frame_for_image = cv2.imread(path)
            if self.loaded_frame_for_image is None:
                messagebox.showerror("Error", f"Failed to load image: {path}")
                self.loaded_path = None
                self.file_label.config(text="No file selected")
            else:
                self.status_label.config(text="Image loaded. Press Submit to process.")
        else:
            self.loaded_frame_for_image = None # Clear any previous image
            self.status_label.config(text="Video loaded. Press Submit to process.")


    def submit_choice(self):
        if not self.loaded_path:
            messagebox.showwarning("No File", "Please open an image or video file first.")
            return
        
        if not self.detector_module or not self.model_objects.get('detector') or not self.model_objects.get('ocr'):
            messagebox.showwarning("Models Not Loaded", "Models are not loaded. Please select a model combination.")
            # Attempt to reload models for current choice
            self.on_model_choice_change()
            if not self.detector_module or not self.model_objects.get('detector') or not self.model_objects.get('ocr'):
                 messagebox.showerror("Model Error", "Failed to load models. Cannot proceed.")
                 return

        self.status_label.config(text="Processing...")
        self.root.update_idletasks()

        try:
            if self.is_video:
                self.process_video_file(self.loaded_path)
            else:
                if self.loaded_frame_for_image is not None:
                    image_out = self.process_frame(self.loaded_frame_for_image.copy())
                    self.show_processed_output(image_out, "Processed Image")
                else:
                    messagebox.showerror("Error", "No image data to process.")
            self.status_label.config(text="Processing complete. Ready for new file or choice.")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred during processing:\n{e}")
            self.status_label.config(text="Error during processing.")
            print(f"Processing error: {e}")


    def process_video_file(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Video Error", f"Could not open video: {video_path}")
            return

        # Create a new Toplevel window for video display
        video_window = tk.Toplevel(self.root)
        video_window.title("Video Processing Output")
        video_label = tk.Label(video_window)
        video_label.pack()

        # Keep track of the Toplevel window to destroy it if closed
        self.active_video_window = video_window
        
        def play_video_frame():
            if not self.active_video_window or not self.active_video_window.winfo_exists(): # Check if window was closed
                cap.release()
                return

            ret, frame = cap.read()
            if not ret:
                cap.release()
                if self.active_video_window and self.active_video_window.winfo_exists():
                    self.active_video_window.destroy()
                self.active_video_window = None
                self.status_label.config(text="Video processing finished.")
                return

            try:
                frame_out = self.process_frame(frame) # Process the current frame
                img_pil = Image.fromarray(cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB))
                
                # Resize if too large for screen (optional, good for large videos)
                max_display_width = self.root.winfo_screenwidth() * 0.8
                max_display_height = self.root.winfo_screenheight() * 0.8
                img_pil.thumbnail((max_display_width, max_display_height), Image.LANCZOS)

                img_tk = ImageTk.PhotoImage(image=img_pil)

                video_label.configure(image=img_tk)
                video_label.image = img_tk # Keep a reference!
                
                # Call next frame
                self.active_video_window.after(30, play_video_frame) # Adjust delay as needed
            except Exception as e:
                print(f"Error processing video frame: {e}")
                cap.release()
                if self.active_video_window and self.active_video_window.winfo_exists():
                    self.active_video_window.destroy()
                self.active_video_window = None
                messagebox.showerror("Video Processing Error", f"Error during video processing: {e}")
                self.status_label.config(text="Error during video processing.")
                return


        play_video_frame() # Start the video loop

    def show_processed_output(self, frame, title="Output"):
        top = tk.Toplevel(self.root)
        top.title(title)
        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Resize if too large for screen (optional)
        max_display_width = self.root.winfo_screenwidth() * 0.8
        max_display_height = self.root.winfo_screenheight() * 0.8
        img_pil.thumbnail((max_display_width, max_display_height), Image.LANCZOS)

        img_tk = ImageTk.PhotoImage(image=img_pil)
        label = tk.Label(top, image=img_tk)
        label.image = img_tk # Keep a reference!
        label.pack()

if __name__ == '__main__':
    root = tk.Tk()
    app = CarPlateApp(root)
    root.mainloop()