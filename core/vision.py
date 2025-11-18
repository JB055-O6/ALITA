"""Vision utilities for Alita.

Enhanced vision system with:
- Camera capture and screen capture
- Face detection and facial expression analysis
- Gesture recognition (hands)
- UI element detection with YOLOv8
- OCR with PaddleOCR
- Screen change detection
- Presence detection

This is intentionally defensive: missing optional libraries are handled
gracefully so the rest of the system keeps working.
"""
from typing import Optional, List, Tuple, Dict, Any
import threading
import time
import logging
from pathlib import Path
import cv2
import numpy as np
from collections import deque

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import mss
except Exception:
    mss = None

try:
    import torch
except Exception:
    torch = None


class VisionSystem:
    """Enhanced camera capture + perception system.

    Methods:
    - start(): begin camera thread
    - stop(): stop camera thread
    - read_frame(): return latest BGR frame (numpy array) or None
    - screen_capture(): capture current screen
    - detect_faces(frame): return list of face detections with expressions
    - detect_gestures(frame): return hand gestures
    - detect_ui_elements(frame): return UI elements (buttons, inputs, etc.)
    - ocr_frame(frame): extract text using PaddleOCR
    - detect_screen_change(): detect if screen has changed
    - detect_presence(): detect if user is present
    - capture_image(path): save current frame to disk
    """

    def __init__(self, config: Optional[dict] = None, camera_index: int = 0):
        self.config = config or {}
        self.camera_index = camera_index
        self._cap = None
        self._running = False
        self._frame = None
        self._thread: Optional[threading.Thread] = None
        
        # Screen capture
        self._screen_capturer = None
        self._last_screen = None
        self._screen_change_threshold = 0.1  # 10% change
        
        # Presence detection
        self._last_face_time = None
        self._presence_timeout = 5.0  # seconds
        
        # Frame history for change detection
        self._frame_history = deque(maxlen=30)  # 1 second at 30 FPS

        # Initialize MediaPipe solutions
        self._mp_face = None
        self._mp_face_mesh = None
        self._mp_hands = None
        self._init_mediapipe()
        
        # Initialize OCR
        self._ocr = None
        self._init_ocr()
        
        # Initialize YOLO for UI detection
        self._yolo = None
        self._init_yolo()

        # Fallback face detector
        self._haar_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._haar_cascade = None
        
        logging.info("Vision System initialized")
    
    def _init_mediapipe(self):
        """Initialize MediaPipe solutions."""
        if mp is None:
            logging.warning("MediaPipe not available")
            return
        
        try:
            # Face detection
            self._mp_face = mp.solutions.face_detection.FaceDetection(
                min_detection_confidence=0.5
            )
            
            # Face mesh for expressions
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=4,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Hand tracking for gestures
            self._mp_hands = mp.solutions.hands.Hands(
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            logging.info("MediaPipe initialized successfully")
        except Exception as e:
            logging.error(f"MediaPipe initialization failed: {str(e)}")
    
    def _init_ocr(self):
        """Initialize PaddleOCR with advanced features."""
        if PaddleOCR is None:
            logging.warning("PaddleOCR not available, falling back to pytesseract")
            return
        
        try:
            # Initialize PaddleOCR with optimized settings
            self._ocr = PaddleOCR(
                use_angle_cls=True,  # Detect text orientation
                lang='en',  # Primary language (supports 180+ languages)
                show_log=False,
                use_gpu=torch.cuda.is_available() if torch else False,
                det_db_thresh=0.3,  # Detection threshold
                det_db_box_thresh=0.5,  # Box threshold
                rec_batch_num=6,  # Batch size for recognition
            )
            logging.info("PaddleOCR initialized with GPU acceleration")
        except Exception as e:
            logging.error(f"PaddleOCR initialization failed: {str(e)}")
            # Fallback to EasyOCR if available
            try:
                import easyocr
                self._ocr_fallback = easyocr.Reader(['en'], gpu=torch.cuda.is_available() if torch else False)
                logging.info("EasyOCR fallback initialized")
            except Exception:
                logging.warning("No OCR engines available")
    
    def _init_yolo(self):
        """Initialize YOLO for UI element detection."""
        if YOLO is None:
            logging.warning("YOLO not available")
            return
        
        try:
            # Use YOLOv8-nano for speed
            model_path = Path("models/yolov8n.pt")
            if model_path.exists():
                self._yolo = YOLO(str(model_path))
                logging.info("YOLO initialized successfully")
            else:
                logging.warning(f"YOLO model not found at {model_path}")
        except Exception as e:
            logging.error(f"YOLO initialization failed: {str(e)}")
    
    def _init_screen_capture(self):
        """Initialize screen capture."""
        if mss is None:
            logging.warning("mss not available for screen capture")
            return
        
        try:
            self._screen_capturer = mss.mss()
            logging.info("Screen capture initialized")
        except Exception as e:
            logging.error(f"Screen capture initialization failed: {str(e)}")

    def start(self) -> None:
        if self._running:
            return
        self._cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not self._cap or not self._cap.isOpened():
            # Try without CAP_DSHOW
            self._cap = cv2.VideoCapture(self.camera_index)
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self) -> None:
        while self._running and self._cap is not None:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            self._frame = frame
            time.sleep(0.01)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None

    def read_frame(self) -> Optional[np.ndarray]:
        return None if self._frame is None else self._frame.copy()

    def screen_capture(self) -> Optional[np.ndarray]:
        """Capture current screen.
        
        Returns:
            BGR numpy array of screen or None
        """
        if self._screen_capturer is None:
            self._init_screen_capture()
        
        if self._screen_capturer is None:
            return None
        
        try:
            # Capture primary monitor
            monitor = self._screen_capturer.monitors[1]
            screenshot = self._screen_capturer.grab(monitor)
            
            # Convert to numpy array
            img = np.array(screenshot)
            
            # Convert BGRA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Store for change detection
            self._last_screen = img.copy()
            
            return img
        except Exception as e:
            logging.error(f"Screen capture failed: {str(e)}")
            return None
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces with facial expression analysis.

        Returns list of dicts with:
        - bbox: (xmin, ymin, w, h)
        - score: confidence score
        - expression: facial expression if available
        - landmarks: facial landmarks if available
        """
        if frame is None:
            return []
        
        h, w = frame.shape[:2]
        faces = []

        # Try MediaPipe Face Mesh for detailed analysis
        if self._mp_face_mesh is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face_mesh.process(rgb)
                
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Get bounding box from landmarks
                        landmarks = face_landmarks.landmark
                        x_coords = [lm.x for lm in landmarks]
                        y_coords = [lm.y for lm in landmarks]
                        
                        xmin = int(min(x_coords) * w)
                        ymin = int(min(y_coords) * h)
                        xmax = int(max(x_coords) * w)
                        ymax = int(max(y_coords) * h)
                        
                        # Analyze expression
                        expression = self._analyze_expression(landmarks)
                        
                        faces.append({
                            "bbox": (xmin, ymin, xmax - xmin, ymax - ymin),
                            "score": 1.0,
                            "expression": expression,
                            "landmarks": [(lm.x * w, lm.y * h) for lm in landmarks]
                        })
                    
                    # Update presence detection
                    self._last_face_time = time.time()
                    return faces
            except Exception as e:
                logging.error(f"Face mesh detection failed: {str(e)}")

        # Fallback to basic face detection
        if self._mp_face is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._mp_face.process(rgb)
                
                if results and results.detections:
                    for det in results.detections:
                        bbox = det.location_data.relative_bounding_box
                        xmin = int(bbox.xmin * w)
                        ymin = int(bbox.ymin * h)
                        bw = int(bbox.width * w)
                        bh = int(bbox.height * h)
                        score = float(det.score[0]) if det.score else 0.0
                        
                        faces.append({
                            "bbox": (xmin, ymin, bw, bh),
                            "score": score,
                            "expression": "neutral",
                            "landmarks": None
                        })
                    
                    self._last_face_time = time.time()
                    return faces
            except Exception:
                pass

        # Final fallback to Haar cascade
        if self._haar_cascade is not None:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self._haar_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, w_, h_) in rects:
                    faces.append({
                        "bbox": (int(x), int(y), int(w_), int(h_)),
                        "score": 1.0,
                        "expression": "neutral",
                        "landmarks": None
                    })
                
                if faces:
                    self._last_face_time = time.time()
            except Exception:
                pass

        return faces
    
    def _analyze_expression(self, landmarks) -> str:
        """Analyze facial expression from landmarks.
        
        Simple heuristic-based expression detection.
        """
        try:
            # Get key landmark indices (MediaPipe Face Mesh)
            # Mouth corners: 61, 291
            # Mouth top/bottom: 13, 14
            # Eyebrows: 70, 300
            
            mouth_left = landmarks[61]
            mouth_right = landmarks[291]
            mouth_top = landmarks[13]
            mouth_bottom = landmarks[14]
            eyebrow_left = landmarks[70]
            eyebrow_right = landmarks[300]
            
            # Calculate mouth openness
            mouth_height = abs(mouth_top.y - mouth_bottom.y)
            mouth_width = abs(mouth_left.x - mouth_right.x)
            
            # Calculate eyebrow height (relative to eyes)
            eye_left = landmarks[33]
            eye_right = landmarks[263]
            eyebrow_height = (eyebrow_left.y + eyebrow_right.y) / 2
            eye_height = (eye_left.y + eye_right.y) / 2
            
            # Simple expression classification
            if mouth_height > 0.03:  # Mouth open
                return "surprised"
            elif mouth_width / mouth_height > 3.0:  # Wide smile
                return "happy"
            elif eyebrow_height < eye_height - 0.02:  # Raised eyebrows
                return "surprised"
            elif eyebrow_height > eye_height + 0.01:  # Lowered eyebrows
                return "angry"
            else:
                return "neutral"
        except Exception:
            return "neutral"
    
    def detect_gestures(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect hand gestures.
        
        Returns list of dicts with:
        - hand: 'left' or 'right'
        - gesture: gesture name
        - landmarks: hand landmarks
        - confidence: detection confidence
        """
        if frame is None or self._mp_hands is None:
            return []
        
        gestures = []
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_hands.process(rgb)
            
            if results and results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get hand label
                    hand_label = "unknown"
                    if results.multi_handedness:
                        hand_label = results.multi_handedness[idx].classification[0].label.lower()
                    
                    # Analyze gesture
                    gesture = self._analyze_gesture(hand_landmarks.landmark)
                    
                    gestures.append({
                        "hand": hand_label,
                        "gesture": gesture,
                        "landmarks": [
                            (lm.x, lm.y, lm.z) 
                            for lm in hand_landmarks.landmark
                        ],
                        "confidence": 1.0
                    })
            
            return gestures
        except Exception as e:
            logging.error(f"Gesture detection failed: {str(e)}")
            return []
    
    def _analyze_gesture(self, landmarks) -> str:
        """Analyze hand gesture from landmarks.
        
        Simple heuristic-based gesture detection.
        """
        try:
            # Get fingertip and base positions
            thumb_tip = landmarks[4]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]
            
            wrist = landmarks[0]
            
            # Check if fingers are extended (tip higher than base)
            thumb_up = thumb_tip.y < landmarks[3].y
            index_up = index_tip.y < landmarks[6].y
            middle_up = middle_tip.y < landmarks[10].y
            ring_up = ring_tip.y < landmarks[14].y
            pinky_up = pinky_tip.y < landmarks[18].y
            
            # Gesture classification
            if thumb_up and not any([index_up, middle_up, ring_up, pinky_up]):
                return "thumbs_up"
            elif all([index_up, middle_up, ring_up, pinky_up]):
                return "open_hand"
            elif index_up and middle_up and not any([ring_up, pinky_up]):
                return "peace"
            elif index_up and not any([middle_up, ring_up, pinky_up]):
                return "pointing"
            elif not any([thumb_up, index_up, middle_up, ring_up, pinky_up]):
                return "fist"
            else:
                return "unknown"
        except Exception:
            return "unknown"
    
    def detect_ui_elements(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect UI elements (buttons, inputs, etc.) using YOLO.
        
        Returns list of dicts with:
        - type: element type (button, input, etc.)
        - bbox: (x, y, w, h)
        - confidence: detection confidence
        """
        if frame is None or self._yolo is None:
            return []
        
        try:
            # Run YOLO detection
            results = self._yolo(frame, verbose=False)
            
            elements = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    # Map class to UI element type
                    element_type = self._map_class_to_ui_element(cls)
                    
                    elements.append({
                        "type": element_type,
                        "bbox": (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                        "confidence": conf
                    })
            
            return elements
        except Exception as e:
            logging.error(f"UI element detection failed: {str(e)}")
            return []
    
    def _map_class_to_ui_element(self, cls: int) -> str:
        """Map YOLO class to UI element type."""
        # This is a simplified mapping - would need custom training for UI elements
        ui_map = {
            0: "button",
            1: "input",
            2: "text",
            3: "image",
            4: "icon"
        }
        return ui_map.get(cls, "unknown")
    
    def ocr_frame(self, frame: np.ndarray) -> str:
        """Extract text using PaddleOCR (falls back to pytesseract).
        
        Returns extracted text.
        """
        if frame is None:
            return ""
        
        # Try PaddleOCR first
        if self._ocr is not None:
            try:
                result = self._ocr.ocr(frame, cls=True)
                
                # Extract text from results
                text_lines = []
                if result and result[0]:
                    for line in result[0]:
                        if line[1]:
                            text_lines.append(line[1][0])
                
                return "\n".join(text_lines)
            except Exception as e:
                logging.error(f"PaddleOCR failed: {str(e)}")
        
        # Fallback to pytesseract
        try:
            import pytesseract
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray)
            return text.strip()
        except Exception:
            return ""
    
    def detect_screen_change(self) -> bool:
        """Detect if screen has changed significantly.
        
        Returns True if screen changed more than threshold.
        """
        current_screen = self.screen_capture()
        
        if current_screen is None or self._last_screen is None:
            return False
        
        try:
            # Resize for faster comparison
            current_small = cv2.resize(current_screen, (320, 240))
            last_small = cv2.resize(self._last_screen, (320, 240))
            
            # Calculate difference
            diff = cv2.absdiff(current_small, last_small)
            diff_percent = np.sum(diff) / (diff.size * 255)
            
            return diff_percent > self._screen_change_threshold
        except Exception as e:
            logging.error(f"Screen change detection failed: {str(e)}")
            return False
    
    def detect_presence(self) -> bool:
        """Detect if user is present (face detected recently).
        
        Returns True if face detected within timeout period.
        """
        if self._last_face_time is None:
            return False
        
        time_since_face = time.time() - self._last_face_time
        return time_since_face < self._presence_timeout
    
    def capture_image(self, path: str) -> bool:
        """Save current frame to disk."""
        frame = self.read_frame()
        if frame is None:
            return False
        try:
            cv2.imwrite(str(path), frame)
            return True
        except Exception:
            return False
