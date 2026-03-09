import cv2
import sqlite3
import os
import json
import requests
import time
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from collections import deque
import threading
import torch
from ultralytics import YOLO
import face_recognition
import pickle
import winsound  # For Windows beep sound

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('live_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suspicious behaviors database
suspicious_behaviors = {
    "Phone or Device Related": [
        "cell phone", "mobile phone", "phone", "smartphone",
        "earpiece", "bluetooth", "headphones", "earbuds",
        "smartwatch", "watch", "electronic device"
    ],
    "Communication or Whispering": [
        "whispering", "talking", "communication", "conversation",
        "mouth_movement", "lip_movement", "gesturing"
    ],
    "Body-Language or Behavioral Cues": [
        "looking_around", "peeking", "copying", "cheating_gesture",
        "fidgeting", "nervous_behavior", "suspicious_movement"
    ],
    "Unauthorized Materials": [
        "book", "paper", "notes", "hidden_material",
        "written_notes", "cheat_sheet", "formula_sheet"
    ],
    "Collusion or Group Tactics": [
        "passing_notes", "exchanging_items", "signaling",
        "group_cheating", "collusion"
    ]
}

class ComprehensiveExamSystem:
    def __init__(self):
        # Configuration
        self.db_file = "exam_hall.db"
        self.faces_dir = "student_faces"
        self.encodings_dir = "face_encodings"
        self.alerts_dir = "monitoring_alerts"
        self.roboflow_api_key = "YOUR_ROBOFLOW_API_KEY"
        self.roboflow_workspace = "YOUR_WORKSPACE"
        self.roboflow_project = "YOUR_PROJECT"
        self.roboflow_version = "1"
        
        # Model paths
        self.yolo_model_path = "best.pt"
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        self.target_fps = 10
        
        # Detection thresholds
        self.confidence_threshold = 0.4  # Lowered for more detections
        self.face_recognition_threshold = 0.6
        self.alert_cooldown = 10  # Reduced cooldown for more alerts
        self.beep_cooldown = 2    # Seconds between beeps during continuous alert
        
        # Monitoring state
        self.is_monitoring = False
        self.last_alert_time = 0
        self.last_beep_time = 0
        self.continuous_beep_active = False
        self.continuous_beep_start_time = 0
        self.known_face_encodings = {}
        self.student_data = {}
        self.detection_history = deque(maxlen=50)  # Increased history
        self.registered_faces = {}
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Behavior tracking
        self.suspicious_behavior_count = 0
        self.current_cheating_incident = None
        
        # Initialize components
        self.setup_database()
        self.setup_directories()
        self.load_student_data()
        self.load_face_encodings()
        self.setup_models()
        self.load_registered_faces()
    
    def play_alert_beep(self, continuous=False):
        """Play beep sound for cheating alert"""
        current_time = time.time()
        
        if continuous:
            # Continuous beeping during active cheating incident
            if current_time - self.last_beep_time >= self.beep_cooldown:
                try:
                    winsound.Beep(1500, 500)  # Higher frequency, shorter duration for continuous
                    self.last_beep_time = current_time
                except Exception as e:
                    print("🔊 BEEP! Cheating in progress!")
        else:
            # Single beep for new detection
            if current_time - self.last_beep_time >= self.beep_cooldown:
                try:
                    winsound.Beep(1000, 1000)  # Standard beep
                    self.last_beep_time = current_time
                    logger.info("🔊 Alert beep played")
                except Exception as e:
                    logger.error(f"Failed to play beep sound: {e}")
                    print("🔊 BEEP! Cheating detected!")
    
    def start_continuous_beeping(self):
        """Start continuous beeping for active cheating incident"""
        if not self.continuous_beep_active:
            self.continuous_beep_active = True
            self.continuous_beep_start_time = time.time()
            print("🚨 CONTINUOUS ALERT: Cheating in progress!")
    
    def stop_continuous_beeping(self):
        """Stop continuous beeping"""
        if self.continuous_beep_active:
            self.continuous_beep_active = False
            duration = time.time() - self.continuous_beep_start_time
            print(f"🛑 Continuous alert ended. Duration: {duration:.1f} seconds")
    
    def setup_database(self):
        """Create database with face storage"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                hand TEXT NOT NULL,
                finger TEXT NOT NULL,
                quality_score INTEGER NOT NULL,
                minutiae_points INTEGER NOT NULL,
                face_image_path TEXT,
                has_face_data BOOLEAN DEFAULT FALSE,
                registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verification_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                verification_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,
                match_score INTEGER,
                quality_diff INTEGER,
                minutiae_diff INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cheating_incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT,
                incident_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                behavior_type TEXT NOT NULL,
                confidence REAL,
                evidence_path TEXT,
                duration_seconds REAL,
                severity TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"✅ Database setup complete! File: {self.db_file}")
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.alerts_dir, exist_ok=True)
        os.makedirs(self.encodings_dir, exist_ok=True)
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
        logger.info("Directories setup complete")
    
    def load_student_data(self):
        """Load student data from fingerprint database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT student_id, name, hand, finger, quality_score, minutiae_points, 
                       face_image_path, has_face_data, registration_date
                FROM students
            ''')
            
            students = cursor.fetchall()
            
            for student in students:
                student_id, name, hand, finger, quality, minutiae, face_path, has_face, reg_date = student
                self.student_data[student_id] = {
                    'name': name,
                    'hand': hand,
                    'finger': finger,
                    'quality_score': quality,
                    'minutiae_points': minutiae,
                    'face_image_path': face_path,
                    'has_face_data': bool(has_face),
                    'registration_date': reg_date,
                    'current_status': 'absent',
                    'last_seen': None,
                    'violations': 0,
                    'cheating_incidents': 0
                }
            
            conn.close()
            logger.info(f"Loaded {len(self.student_data)} students from database")
            
        except Exception as e:
            logger.error(f"Failed to load student data: {e}")
    
    def load_face_encodings(self):
        """Load or generate face encodings from registered students"""
        logger.info("Loading face encodings...")
        
        # Load pre-computed encodings
        for encoding_file in Path(self.encodings_dir).glob("*.pkl"):
            student_id = encoding_file.stem.replace("_encoding", "")
            try:
                with open(encoding_file, 'rb') as f:
                    self.known_face_encodings[student_id] = pickle.load(f)
                logger.debug(f"Loaded encoding for {student_id}")
            except Exception as e:
                logger.error(f"Failed to load encoding for {student_id}: {e}")
        
        # Generate missing encodings
        for student_id, data in self.student_data.items():
            if data['has_face_data'] and student_id not in self.known_face_encodings:
                face_path = data['face_image_path']
                if face_path and os.path.exists(face_path):
                    encoding = self.generate_face_encoding(face_path)
                    if encoding is not None:
                        self.known_face_encodings[student_id] = encoding
                        # Save encoding for future use
                        encoding_path = Path(self.encodings_dir) / f"{student_id}_encoding.pkl"
                        with open(encoding_path, 'wb') as f:
                            pickle.dump(encoding, f)
                        logger.info(f"Generated and saved encoding for {student_id}")
        
        logger.info(f"Total face encodings loaded: {len(self.known_face_encodings)}")
    
    def generate_face_encoding(self, image_path):
        """Generate face encoding from image"""
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            return encodings[0] if encodings else None
        except Exception as e:
            logger.error(f"Failed to generate encoding for {image_path}: {e}")
            return None
    
    def setup_models(self):
        """Setup YOLO model and other ML models"""
        try:
            # Load YOLO model for cheating detection
            if os.path.exists(self.yolo_model_path):
                self.yolo_model = YOLO(self.yolo_model_path)
                logger.info(f"Loaded fine-tuned YOLO model: {self.yolo_model_path}")
            else:
                # Fallback to a standard YOLO model
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("Loaded default YOLO model")
            
            # Set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using {device.upper()} for inference")
                
        except Exception as e:
            logger.error(f"Failed to setup YOLO model: {e}")
            self.yolo_model = None
    
    def load_registered_faces(self):
        """Load all registered student faces from student_faces folder"""
        print("Loading registered faces from student_faces folder...")
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            print(f"Created {self.faces_dir} directory")
            print("Please add student face images to this folder.")
            return
            
        loaded_count = 0
        for file in os.listdir(self.faces_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Extract student ID from filename (format: ID_NAME.jpg)
                filename_no_ext = os.path.splitext(file)[0]
                parts = filename_no_ext.split('_')
                
                if len(parts) >= 2:
                    student_id = parts[0]  # First part is ID
                    student_name = parts[1]  # Second part is name
                else:
                    student_id = filename_no_ext
                    student_name = "Unknown"
                
                img_path = os.path.join(self.faces_dir, file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Preprocess image for better comparison
                    processed_img = self.preprocess_face(img)
                    if processed_img is not None:
                        self.registered_faces[student_id] = {
                            'image': processed_img,
                            'name': student_name,
                            'filename': file
                        }
                        loaded_count += 1
                        print(f"✓ Loaded: {student_id} - {student_name}")
                    else:
                        print(f"✗ Could not detect face in: {file}")
                else:
                    print(f"✗ Failed to load image: {file}")
        
        print(f"Total registered faces loaded: {loaded_count}")
        
        if loaded_count == 0:
            print("\n⚠️  No faces found! Please add images to student_faces folder.")
            print("Image naming format: STUDENTID_NAME.jpg")
            print("Example: 1HK22AI013_HARINI.jpg")
    
    def preprocess_face(self, img):
        """Detect and preprocess face from image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            print("    No face detected in image, using entire image")
            # If no face detected, use the entire image but resize
            face_standard = cv2.resize(img, (200, 200))
        else:
            # Use the first detected face
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            # Resize to standard size
            face_standard = cv2.resize(face_roi, (200, 200))
        
        # Apply histogram equalization for better contrast
        face_ycrcb = cv2.cvtColor(face_standard, cv2.COLOR_BGR2YCrCb)
        face_ycrcb[:,:,0] = cv2.equalizeHist(face_ycrcb[:,:,0])
        face_processed = cv2.cvtColor(face_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return face_processed
    
    def extract_face_from_frame(self, frame):
        """Extract face region from webcam frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, []
            
        # Return the largest face
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        
        # Extract and preprocess face
        face_roi = frame[y:y+h, x:x+w]
        face_processed = self.preprocess_face(face_roi)
        
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, "Face Detected", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return face_processed, faces
    
    def compare_faces_improved(self, img1, img2, threshold=0.5):
        """Improved face comparison using multiple methods"""
        try:
            # Method 1: Histogram comparison (HSV)
            img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
            img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            
            hist1 = cv2.calcHist([img1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            hist2 = cv2.calcHist([img2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
            
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            
            hist_score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            
            # Method 2: Structural Similarity
            try:
                from skimage.metrics import structural_similarity as ssim
                gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                ssim_score = ssim(gray1, gray2)
            except ImportError:
                ssim_score = 0.3  # Lower default if not available
            
            # Method 3: Template Matching (simple but effective)
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)
            template_score = np.max(result)
            
            # Combined score (weighted average)
            combined_score = (hist_score * 0.5 + ssim_score * 0.3 + template_score * 0.2)
            
            # Debug output
            print(f"  Scores - Hist: {hist_score:.3f}, SSIM: {ssim_score:.3f}, Template: {template_score:.3f}, Combined: {combined_score:.3f}")
            
            return combined_score > threshold, combined_score
            
        except Exception as e:
            print(f"Comparison error: {e}")
            return False, 0.0
    
    def get_student_info(self, student_id):
        """Get student information from database or filename"""
        try:
            # First try to get from database
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0], "N/A"
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        
        # Fallback to filename data
        if student_id in self.registered_faces:
            return self.registered_faces[student_id]['name'], "N/A"
        
        return ("Unknown", "N/A")
    
    # ==================== REGISTRATION METHODS ====================
    
    def capture_face_with_camera(self, student_id, name):
        """Capture face image using laptop camera"""
        print("\n📷 FACE CAPTURE WITH CAMERA")
        print("=" * 50)
        print("INSTRUCTIONS:")
        print("1. Look directly at the camera")
        print("2. Make sure your face is well-lit")
        print("3. Press SPACEBAR to capture photo")
        print("4. Press 'q' to quit without saving")
        print("-" * 50)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access camera")
            print("💡 Make sure your laptop camera is not being used by another application")
            return None
        
        print("✅ Camera opened successfully")
        print("🔍 Looking for camera...")
        
        face_captured = False
        captured_frame = None
        
        while True:
            # Read frame from camera
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Cannot read from camera")
                break
            
            # Display the camera feed
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACEBAR to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face Capture - Press SPACE to capture, Q to quit', display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # SPACEBAR to capture
                captured_frame = frame.copy()
                face_captured = True
                print("✅ Photo captured!")
                break
            elif key == ord('q'):  # 'q' to quit
                print("❌ Capture cancelled")
                break
        
        # Release camera and close window
        cap.release()
        cv2.destroyAllWindows()
        
        if not face_captured or captured_frame is None:
            return None
        
        # Save the captured image
        image_filename = f"{student_id}_{name.replace(' ', '_')}.jpg"
        image_path = os.path.join(self.faces_dir, image_filename)
        
        try:
            # Save the image
            success = cv2.imwrite(image_path, captured_frame)
            if success:
                print(f"✅ Face image saved: {image_path}")
                return image_path
            else:
                print("❌ Error: Could not save image")
                return None
        except Exception as e:
            print(f"❌ Error saving image: {e}")
            return None
    
    def test_camera(self):
        """Test if camera is working"""
        print("\n📸 TESTING CAMERA...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Camera not accessible")
            return False
        
        print("✅ Camera is accessible")
        print("🔍 Showing camera preview for 3 seconds...")
        
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 3:
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Camera Test - Close window after 3 seconds', frame)
                cv2.waitKey(1)
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Camera test completed successfully!")
        return True
    
    def register_student(self):
        """Register new student with fingerprint and face capture"""
        print("\n" + "="*50)
        print("🎓 STUDENT REGISTRATION")
        print("="*50)
        
        student_id = input("Enter Student ID: ").strip()
        name = input("Enter Student Name: ").strip()
        
        # Fingerprint registration
        print("\n📋 FINGER SELECTION:")
        print("Available fingers:")
        print("RIGHT: THUMB, INDEX, MIDDLE, RING, LITTLE")
        print("LEFT: THUMB, INDEX, MIDDLE, RING, LITTLE")
        print("-" * 40)
        
        hand = input("Which hand? (RIGHT/LEFT): ").strip().upper()
        finger = input("Which finger? (THUMB/INDEX/MIDDLE/RING/LITTLE): ").strip().upper()
        
        print("\n📊 FINGERPRINT CAPTURE:")
        print("For better accuracy, we'll take 3 samples!")
        print("-" * 40)
        
        samples = []
        for i in range(3):
            print(f"\nSample {i+1}:")
            print("1. Capture fingerprint using Mantra software")
            print("2. Enter the quality score and minutiae points")
            
            try:
                quality = int(input("Quality score (0-100): ").strip())
                minutiae = int(input("Number of minutiae points: ").strip())
                samples.append((quality, minutiae))
                print(f"✅ Sample {i+1} recorded: Q={quality}, M={minutiae}")
            except ValueError:
                print("❌ Please enter valid numbers!")
                return
        
        # Calculate averages
        avg_quality = sum(s[0] for s in samples) // len(samples)
        avg_minutiae = sum(s[1] for s in samples) // len(samples)
        
        print(f"\n📈 Fingerprint averages calculated:")
        print(f"Quality: {avg_quality}, Minutiae: {avg_minutiae}")
        
        # Face capture with camera
        print("\n" + "="*50)
        face_image_path = self.capture_face_with_camera(student_id, name)
        
        if face_image_path is None:
            print("❌ Face capture failed or was cancelled.")
            choice = input("Continue without face data? (y/n): ").lower()
            if choice != 'y':
                return
            has_face = False
            face_image_path = None
        else:
            has_face = True
        
        # Save to database
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO students 
                (student_id, name, hand, finger, quality_score, minutiae_points, 
                 face_image_path, has_face_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (student_id, name, hand, finger, avg_quality, avg_minutiae, 
                  face_image_path, has_face))
            
            conn.commit()
            conn.close()
            
            print("\n✅ Student registered successfully!")
            print(f"   Student ID: {student_id}")
            print(f"   Name: {name}")
            print(f"   Finger: {hand} {finger}")
            print(f"   Avg Quality: {avg_quality}, Avg Minutiae: {avg_minutiae}")
            print(f"   Face Data Stored: {'✅' if has_face else '❌'}")
            if has_face:
                print(f"   Face Image: {face_image_path}")
            
        except sqlite3.IntegrityError:
            print("❌ Student ID already exists!")
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    def add_face_to_existing_student(self):
        """Add face data to an already registered student using camera"""
        print("\n" + "="*50)
        print("📷 ADD FACE DATA TO EXISTING STUDENT")
        print("="*50)
        
        student_id = input("Enter Student ID: ").strip()
        
        # Check if student exists
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        cursor.execute('SELECT name, has_face_data FROM students WHERE student_id = ?', (student_id,))
        result = cursor.fetchone()
        
        if not result:
            print("❌ Student not found!")
            conn.close()
            return
        
        name, has_face_data = result
        conn.close()
        
        print(f"👤 Student Found: {name}")
        
        if has_face_data:
            print("⚠️  This student already has face data.")
            choice = input("Overwrite existing face data? (y/n): ").lower()
            if choice != 'y':
                return
        
        # Capture new face data with camera
        face_image_path = self.capture_face_with_camera(student_id, name)
        
        if face_image_path is None:
            print("❌ Face capture failed.")
            return
        
        # Update database
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE students 
                SET face_image_path = ?, has_face_data = ?
                WHERE student_id = ?
            ''', (face_image_path, True, student_id))
            
            conn.commit()
            conn.close()
            
            print("✅ Face data added successfully!")
            print(f"   Student: {name} ({student_id})")
            print(f"   Face Image: {face_image_path}")
            
        except Exception as e:
            print(f"❌ Database error: {e}")
    
    def view_face_data_info(self):
        """View information about stored face data"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT student_id, name, has_face_data, face_image_path 
            FROM students 
            ORDER BY student_id
        ''')
        students = cursor.fetchall()
        
        print("\n📊 FACE DATA INFORMATION:")
        print("-" * 80)
        if not students:
            print("No students registered yet.")
        else:
            face_count = 0
            for student in students:
                student_id, name, has_face, image_path = student
                face_status = "✅" if has_face else "❌"
                image_file = os.path.basename(image_path) if image_path else "N/A"
                print(f"ID: {student_id} | Name: {name:15} | Face: {face_status} | Image: {image_file}")
                if has_face:
                    face_count += 1
            
            print(f"\n📈 Summary: {face_count}/{len(students)} students have face data")
        
        conn.close()
    
    def verify_student(self):
        """Verify student using fingerprint only"""
        print("\n" + "="*50)
        print("📝 EXAM HALL VERIFICATION (Fingerprint Only)")
        print("="*50)
        
        student_id = input("Enter Student ID: ").strip()
        
        # Get student data from database
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, hand, finger, quality_score, minutiae_points, has_face_data 
            FROM students WHERE student_id = ?
        ''', (student_id,))
        
        result = cursor.fetchone()
        
        if not result:
            print("❌ Student not found! Please check Student ID.")
            conn.close()
            return False
        
        stored_name, stored_hand, stored_finger, stored_quality, stored_minutiae, has_face_data = result
        conn.close()
        
        print(f"\n👤 Student Found: {stored_name}")
        print(f"📋 Registered: {stored_hand} {stored_finger}")
        print(f"   Stored - Quality: {stored_quality}, Minutiae: {stored_minutiae}")
        print(f"   Face Data Available: {'✅' if has_face_data else '❌'}")
        
        print("\n🎯 FINGERPRINT VERIFICATION:")
        print(f"Please scan: {stored_hand} {stored_finger}")
        print("1. Capture fingerprint with Mantra software")
        print("2. Enter new quality score and minutiae points")
        print("-" * 40)
        
        try:
            new_quality = int(input("New quality score: ").strip())
            new_minutiae = int(input("New minutiae points: ").strip())
        except ValueError:
            print("❌ Please enter valid numbers!")
            return False
        
        # Calculate match score
        match_score = self.calculate_match_score(
            stored_quality, stored_minutiae, 
            new_quality, new_minutiae
        )
        
        print(f"🎯 Overall Match Score: {match_score}%")
        
        # Verification
        threshold = 60 if stored_quality < 50 else 70
            
        if match_score >= threshold:
            print("✅ FINGERPRINT VERIFIED!")
            print("🎉 Identity confirmed - Allow entry to exam hall")
            self.log_verification(student_id, "SUCCESS", match_score, 
                                abs(stored_quality - new_quality), 
                                abs(stored_minutiae - new_minutiae))
            return True
        else:
            print("❌ FINGERPRINT MISMATCH!")
            print("🚫 Identity not verified - Please see invigilator")
            self.log_verification(student_id, "FAILED", match_score,
                                abs(stored_quality - new_quality),
                                abs(stored_minutiae - new_minutiae))
            return False
    
    def calculate_match_score(self, stored_quality, stored_minutiae, new_quality, new_minutiae):
        """Calculate intelligent match score"""
        quality_diff = abs(stored_quality - new_quality)
        minutiae_diff = abs(stored_minutiae - new_minutiae)
        
        if quality_diff <= 10:
            quality_score = 40
        elif quality_diff <= 25:
            quality_score = 30
        elif quality_diff <= 40:
            quality_score = 20
        else:
            quality_score = 0
        
        minutiae_ratio = min(stored_minutiae, new_minutiae) / max(stored_minutiae, new_minutiae)
        minutiae_score = minutiae_ratio * 30
        
        range_bonus = 10 if (stored_quality > 50 and new_quality > 50) or (stored_quality <= 50 and new_quality <= 50) else 0
        
        return min(100, int(quality_score + minutiae_score + range_bonus))
    
    def log_verification(self, student_id, status, match_score, quality_diff, minutiae_diff):
        """Log verification attempts"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO verification_log (student_id, status, match_score, quality_diff, minutiae_diff)
            VALUES (?, ?, ?, ?, ?)
        ''', (student_id, status, match_score, quality_diff, minutiae_diff))
        
        conn.commit()
        conn.close()
    
    def view_students(self):
        """View all registered students"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT student_id, name, hand, finger, quality_score, minutiae_points, has_face_data 
            FROM students ORDER BY registration_date
        ''')
        students = cursor.fetchall()
        
        print("\n📊 REGISTERED STUDENTS:")
        print("-" * 80)
        if not students:
            print("No students registered yet.")
        else:
            for student in students:
                student_id, name, hand, finger, quality, minutiae, has_face = student
                quality_status = "🟢" if quality > 60 else "🟡" if quality > 40 else "🔴"
                face_status = "✅" if has_face else "❌"
                print(f"ID: {student_id} | Name: {name:15} | {hand} {finger:8} | {quality_status} Q:{quality:3} M:{minutiae:3} | Face: {face_status}")
        
        conn.close()
    
    # ==================== SEAT VERIFICATION METHODS ====================
    
    def run_seat_verification(self):
        """Main verification loop"""
        if len(self.registered_faces) == 0:
            print("❌ No faces available for verification!")
            print("Please add student face images to the 'student_faces' folder.")
            print("Format: STUDENTID_NAME.jpg (e.g., 1HK22AI013_HARINI.jpg)")
            return
            
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot access camera!")
            return
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n" + "="*50)
        print("ENHANCED SEAT VERIFICATION SYSTEM")
        print("="*50)
        print(f"Loaded {len(self.registered_faces)} students from student_faces folder")
        print("Registered students:")
        for student_id, data in self.registered_faces.items():
            print(f"  - {student_id}: {data['name']}")
        print("\nInstructions:")
        print("- Ensure good lighting and face the camera directly")
        print("- Press 'q' to quit")
        print("- Press 'r' to reload faces")
        print("="*50)
        
        verification_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Extract face from current frame
            detected_face, faces = self.extract_face_from_frame(display_frame)
            
            detected_name = "Unknown"
            detected_seat = "N/A"
            confidence = 0.0
            
            if detected_face is not None:
                # Compare with all registered faces
                best_match_id = None
                best_confidence = 0.0
                
                print(f"\nComparing with {len(self.registered_faces)} registered faces...")
                for student_id, data in self.registered_faces.items():
                    ref_face = data['image']
                    match, conf_score = self.compare_faces_improved(ref_face, detected_face)
                    if match and conf_score > best_confidence:
                        best_confidence = conf_score
                        best_match_id = student_id
                        print(f"  ✓ Potential match: {data['name']} (score: {conf_score:.3f})")
                
                if best_match_id:
                    detected_name, detected_seat = self.get_student_info(best_match_id)
                    confidence = best_confidence
                    verification_count += 1
                    print(f"✅ MATCH FOUND: {detected_name} (Seat: {detected_seat})")
            
            # Display results
            status_text = f"{detected_name} - Seat {detected_seat}"
            if detected_name != "Unknown":
                confidence_text = f"Confidence: {confidence:.2f}"
                cv2.putText(display_frame, confidence_text, (30, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                color = (0, 255, 0)  # Green for verified
            else:
                if len(faces) > 0:
                    status_text = "Unknown - No Match Found"
                color = (0, 0, 255)  # Red for unknown
            
            cv2.putText(display_frame, status_text, (30, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Display face count
            face_count_text = f"Faces detected: {len(faces)}"
            cv2.putText(display_frame, face_count_text, (30, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display registered faces count
            reg_text = f"Registered: {len(self.registered_faces)} students"
            cv2.putText(display_frame, reg_text, (30, 140), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("Enhanced Seat Verification", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Reloading faces...")
                self.registered_faces.clear()
                self.load_registered_faces()
                if len(self.registered_faces) == 0:
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete")
    
    # ==================== ENHANCED LIVE MONITORING METHODS ====================
    
    def detect_cheating_behaviors(self, frame):
        """Detect cheating behaviors using YOLO model with enhanced detection"""
        cheating_detections = []
        
        if self.yolo_model is None:
            return cheating_detections
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=self.confidence_threshold, verbose=False)
            
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        confidence = float(box.conf)
                        class_id = int(box.cls)
                        class_name = self.yolo_model.names[class_id]
                        bbox = box.xyxy[0].tolist()
                        
                        # Enhanced cheating detection for various behaviors
                        detected_behavior = self.classify_behavior(class_name, confidence, bbox, frame)
                        
                        if detected_behavior:
                            cheating_detections.append({
                                'class': class_name,
                                'behavior_type': detected_behavior['type'],
                                'confidence': confidence,
                                'bbox': bbox,
                                'severity': detected_behavior['severity'],
                                'timestamp': datetime.now().isoformat()
                            })
            
        except Exception as e:
            logger.error(f"Cheating detection error: {e}")
        
        return cheating_detections
    
    def classify_behavior(self, class_name, confidence, bbox, frame):
        """Classify detected objects into specific suspicious behaviors"""
        class_name_lower = class_name.lower()
        
        # Phone and device related
        if any(device in class_name_lower for device in ['phone', 'cell', 'mobile', 'smartphone']):
            return {'type': 'phone_usage', 'severity': 'high'}
        
        elif any(device in class_name_lower for device in ['headphone', 'earphone', 'earpiece', 'bluetooth']):
            return {'type': 'communication_device', 'severity': 'high'}
        
        elif any(device in class_name_lower for device in ['watch', 'smartwatch']):
            return {'type': 'smartwatch_usage', 'severity': 'medium'}
        
        # Unauthorized materials
        elif any(material in class_name_lower for material in ['book', 'notebook', 'paper']):
            return {'type': 'unauthorized_materials', 'severity': 'medium'}
        
        # Person behaviors (need additional analysis)
        elif class_name_lower == 'person':
            # Analyze person behavior for suspicious patterns
            behavior_analysis = self.analyze_person_behavior(bbox, frame)
            if behavior_analysis:
                return behavior_analysis
        
        # Electronic devices
        elif any(device in class_name_lower for device in ['laptop', 'tablet', 'ipad', 'electronic']):
            return {'type': 'electronic_device', 'severity': 'high'}
        
        return None
    
    def analyze_person_behavior(self, bbox, frame):
        """Analyze person bounding box for suspicious behaviors"""
        x1, y1, x2, y2 = map(int, bbox)
        person_roi = frame[y1:y2, x1:x2]
        
        # Simple analysis based on position and movement patterns
        frame_center_x = frame.shape[1] // 2
        person_center_x = (x1 + x2) // 2
        
        # Detect if person is looking around (head position)
        if abs(person_center_x - frame_center_x) > frame.shape[1] * 0.3:
            return {'type': 'looking_around', 'severity': 'medium'}
        
        # Detect hand movements near face/mouth (potential whispering)
        # This is a simplified version - real implementation would need pose estimation
        if y2 - y1 < frame.shape[0] * 0.4:  # Person is sitting/leaning
            return {'type': 'suspicious_posture', 'severity': 'low'}
        
        return None
    
    def detect_suspicious_patterns(self, detections, frame):
        """Detect complex suspicious patterns from multiple detections"""
        patterns = []
        current_time = datetime.now()
        
        # Count recent detections by type
        recent_detections = [
            item for item in self.detection_history 
            if current_time - item['timestamp'] < timedelta(minutes=2)
        ]
        
        behavior_counts = {}
        for detection_item in recent_detections:
            for detection in detection_item['detections']:
                behavior_type = detection.get('behavior_type', 'unknown')
                behavior_counts[behavior_type] = behavior_counts.get(behavior_type, 0) + 1
        
        # Pattern 1: Multiple device detections
        if behavior_counts.get('phone_usage', 0) >= 2:
            patterns.append({
                'type': 'repeated_phone_usage',
                'count': behavior_counts['phone_usage'],
                'severity': 'high'
            })
        
        # Pattern 2: Combination of different cheating methods
        unique_cheating_types = len([bt for bt in behavior_counts.keys() if bt != 'unknown'])
        if unique_cheating_types >= 3:
            patterns.append({
                'type': 'multiple_cheating_methods',
                'count': unique_cheating_types,
                'severity': 'high'
            })
        
        # Pattern 3: Continuous suspicious activity
        if len(recent_detections) >= 5:
            patterns.append({
                'type': 'continuous_suspicious_activity',
                'count': len(recent_detections),
                'severity': 'medium'
            })
        
        return patterns
    
    def save_cheating_evidence(self, frame, detection, incident_id):
        """Save detailed evidence of cheating incident"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save image with bounding boxes
        evidence_frame = frame.copy()
        
        # Draw detection bounding box
        bbox = detection.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 0, 255)  # Red for cheating
            cv2.rectangle(evidence_frame, (x1, y1), (x2, y2), color, 3)
            
            label = f"{detection.get('behavior_type', 'unknown')} ({detection.get('confidence', 0):.2f})"
            cv2.putText(evidence_frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Add timestamp and incident info
        info_text = f"Incident {incident_id} - {datetime.now().strftime('%H:%M:%S')}"
        cv2.putText(evidence_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save image
        image_filename = f"cheating_incident_{incident_id}_{timestamp}.jpg"
        image_path = os.path.join(self.alerts_dir, image_filename)
        cv2.imwrite(image_path, evidence_frame)
        
        return image_path
    
    def log_cheating_incident(self, student_id, behavior_type, confidence, evidence_path, severity):
        """Log cheating incident to database"""
        try:
            conn = sqlite3.connect(self.db_file)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO cheating_incidents 
                (student_id, behavior_type, confidence, evidence_path, severity)
                VALUES (?, ?, ?, ?, ?)
            ''', (student_id, behavior_type, confidence, evidence_path, severity))
            
            incident_id = cursor.lastrowid
            
            conn.commit()
            conn.close()
            
            # Update student violations
            if student_id and student_id != "Unknown" and student_id in self.student_data:
                self.student_data[student_id]['violations'] += 1
                self.student_data[student_id]['cheating_incidents'] += 1
                
                # Escalate status based on violations
                violations = self.student_data[student_id]['violations']
                if violations >= 3:
                    self.student_data[student_id]['current_status'] = 'cheating'
                elif violations >= 1:
                    self.student_data[student_id]['current_status'] = 'suspicious'
            
            return incident_id
            
        except Exception as e:
            logger.error(f"Failed to log cheating incident: {e}")
            return None
    
    def send_enhanced_alert(self, alert_data, frame):
        """Send enhanced alert with continuous beeping and evidence saving"""
        # Generate incident ID
        incident_id = int(datetime.now().timestamp())
        
        # Save evidence
        evidence_path = self.save_cheating_evidence(frame, alert_data, incident_id)
        
        # Log to database
        student_id = alert_data.get('student_id', 'unknown')
        behavior_type = alert_data.get('behavior_type', 'unknown')
        confidence = alert_data.get('confidence', 0)
        severity = alert_data.get('severity', 'medium')
        
        db_incident_id = self.log_cheating_incident(student_id, behavior_type, confidence, evidence_path, severity)
        
        # Start continuous beeping for high severity incidents
        if severity == 'high':
            self.start_continuous_beeping()
            self.current_cheating_incident = {
                'id': incident_id,
                'start_time': time.time(),
                'behavior': behavior_type,
                'student_id': student_id
            }
        
        # Log alert
        logger.warning(f"🚨 CHEATING ALERT: {behavior_type} - Student: {student_id} - Confidence: {confidence:.2f}")
        print(f"\n🚨 CHEATING ALERT #{incident_id}")
        print(f"   Behavior: {behavior_type}")
        print(f"   Student: {student_id}")
        print(f"   Confidence: {confidence:.2f}")
        print(f"   Severity: {severity.upper()}")
        print(f"   Evidence: {evidence_path}")
        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
        
        if severity == 'high':
            print("   🔄 CONTINUOUS BEEPING ACTIVATED!")
        
        return incident_id
    
    def recognize_faces(self, frame):
        """Recognize faces in the frame"""
        face_detections = []
        
        if not self.known_face_encodings:
            return face_detections
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find all face locations and encodings
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare with known faces
                matches = face_recognition.compare_faces(
                    list(self.known_face_encodings.values()), 
                    face_encoding,
                    tolerance=self.face_recognition_threshold
                )
                
                face_distances = face_recognition.face_distance(
                    list(self.known_face_encodings.values()), 
                    face_encoding
                )
                
                best_match_index = np.argmin(face_distances) if face_distances.size > 0 else -1
                
                student_id = "Unknown"
                confidence = 0.0
                
                if best_match_index != -1 and matches[best_match_index]:
                    student_id = list(self.known_face_encodings.keys())[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
                
                face_detections.append({
                    'student_id': student_id,
                    'confidence': confidence,
                    'bbox': [left, top, right, bottom],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Update student status
                if student_id != "Unknown" and student_id in self.student_data:
                    self.student_data[student_id]['current_status'] = 'present'
                    self.student_data[student_id]['last_seen'] = datetime.now().isoformat()
        
        except Exception as e:
            logger.error(f"Face recognition error: {e}")
        
        return face_detections
    
    def detect_roboflow_behaviors(self, frame):
        """Detect behaviors using Roboflow API"""
        roboflow_detections = []
        
        if self.roboflow_api_key == "YOUR_ROBOFLOW_API_KEY":
            return roboflow_detections
            
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            image_bytes = buffer.tobytes()
            
            # Roboflow API call
            url = f"https://detect.roboflow.com/{self.roboflow_project}/{self.roboflow_version}"
            params = {
                'api_key': self.roboflow_api_key,
                'confidence': self.confidence_threshold
            }
            
            response = requests.post(
                url,
                params=params,
                data=image_bytes,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                for prediction in result.get('predictions', []):
                    class_name = prediction['class']
                    confidence = prediction['confidence']
                    bbox = [
                        prediction['x'] - prediction['width'] / 2,
                        prediction['y'] - prediction['height'] / 2,
                        prediction['x'] + prediction['width'] / 2,
                        prediction['y'] + prediction['height'] / 2
                    ]
                    
                    roboflow_detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'timestamp': datetime.now().isoformat(),
                        'source': 'roboflow'
                    })
            
        except Exception as e:
            logger.error(f"Roboflow API error: {e}")
        
        return roboflow_detections
    
    def process_frame(self, frame):
        """Process a single frame for enhanced monitoring"""
        all_detections = []
        
        try:
            # 1. Face recognition
            face_detections = self.recognize_faces(frame)
            all_detections.extend(face_detections)
            
            # 2. Enhanced cheating behavior detection
            cheating_detections = self.detect_cheating_behaviors(frame)
            all_detections.extend(cheating_detections)
            
            # 3. Roboflow detection (if API key available)
            roboflow_detections = self.detect_roboflow_behaviors(frame)
            all_detections.extend(roboflow_detections)
            
            # 4. Pattern analysis
            suspicious_patterns = self.detect_suspicious_patterns(all_detections, frame)
            
            # 5. Check for alerts
            current_time = time.time()
            if current_time - self.last_alert_time >= self.alert_cooldown:
                self.check_for_alerts(all_detections, suspicious_patterns, frame)
            
            # 6. Handle continuous beeping
            if self.continuous_beep_active:
                self.play_alert_beep(continuous=True)
                # Auto-stop continuous beeping after 30 seconds
                if current_time - self.continuous_beep_start_time > 30:
                    self.stop_continuous_beeping()
            
            return all_detections, suspicious_patterns
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], []
    
    def check_for_alerts(self, detections, suspicious_patterns, frame):
        """Check if any detections warrant enhanced alerts"""
        current_time = time.time()
        
        try:
            # Check individual cheating detections
            for detection in detections:
                if detection.get('behavior_type') and detection.get('confidence', 0) > 0.3:
                    alert_data = {
                        'type': 'cheating_behavior',
                        'behavior_type': detection['behavior_type'],
                        'severity': detection.get('severity', 'medium'),
                        'confidence': detection.get('confidence', 0),
                        'student_id': detection.get('student_id', 'unknown'),
                        'details': detection
                    }
                    self.send_enhanced_alert(alert_data, frame)
                    self.last_alert_time = current_time
            
            # Check suspicious patterns
            for pattern in suspicious_patterns:
                if pattern['severity'] in ['high', 'medium']:
                    alert_data = {
                        'type': 'suspicious_pattern',
                        'behavior_type': pattern['type'],
                        'severity': pattern['severity'],
                        'count': pattern['count'],
                        'details': pattern
                    }
                    self.send_enhanced_alert(alert_data, frame)
                    self.last_alert_time = current_time
                    
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
    
    def draw_enhanced_detections(self, frame, detections, patterns):
        """Draw enhanced detections on frame with behavior information"""
        # Draw individual detections
        for detection in detections:
            try:
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Choose color based on severity
                    severity = detection.get('severity', 'medium')
                    if severity == 'high':
                        color = (0, 0, 255)  # Red for high severity
                    elif severity == 'medium':
                        color = (0, 165, 255)  # Orange for medium
                    else:
                        color = (0, 255, 255)  # Yellow for low
                    
                    # Choose label based on detection type
                    if detection.get('student_id', 'Unknown') != "Unknown":
                        student_id = detection.get('student_id', 'Unknown')
                        confidence = detection.get('confidence', 0)
                        label = f"Student: {student_id} ({confidence:.2f})"
                    elif detection.get('behavior_type'):
                        behavior_type = detection['behavior_type']
                        confidence = detection.get('confidence', 0)
                        label = f"{behavior_type} ({confidence:.2f})"
                    else:
                        label = "Suspicious Behavior"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Draw label with background for better visibility
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                                (x1 + label_size[0], y1), color, -1)
                    cv2.putText(frame, label, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except Exception as e:
                logger.error(f"Error drawing detection: {e}")
                continue
        
        # Add status information
        status_y = 30
        cv2.putText(frame, f"Active Detections: {len(detections)}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        status_y += 25
        
        if self.continuous_beep_active:
            cv2.putText(frame, "🚨 CONTINUOUS ALERT: CHEATING IN PROGRESS", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            status_y += 25
        
        cv2.putText(frame, f"Suspicious Behaviors: {self.suspicious_behavior_count}", (10, status_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame

    def start_live_monitoring(self):
        """Start the enhanced live monitoring process"""
        logger.info("Starting enhanced live monitoring...")
        self.is_monitoring = True
        self.suspicious_behavior_count = 0
        self.continuous_beep_active = False
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        
        if not cap.isOpened():
            logger.error("Cannot open camera")
            print("❌ Error: Cannot open camera. Please check camera connection.")
            self.is_monitoring = False
            return
        
        logger.info("Camera initialized successfully")
        print("✅ Camera initialized successfully")
        print("🔊 Enhanced monitoring activated!")
        print("📸 Photos will be automatically saved when cheating is detected")
        print("🚨 Continuous beeping for high-severity incidents")
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.is_monitoring:
                ret, frame = cap.read()
                
                if not ret:
                    logger.error("Failed to capture frame")
                    print("❌ Error: Failed to capture frame from camera")
                    break
                
                # Process frame every 3 frames to balance performance and detection
                if frame_count % 3 == 0:
                    try:
                        detections, suspicious_patterns = self.process_frame(frame)
                        
                        # Update behavior count
                        if detections:
                            self.suspicious_behavior_count += len([d for d in detections if d.get('behavior_type')])
                        
                        # Draw enhanced detections on frame
                        display_frame = self.draw_enhanced_detections(frame.copy(), detections, suspicious_patterns)
                        
                        # Show frame
                        cv2.imshow('Enhanced Live Monitoring - Press Q to quit', display_frame)
                    except Exception as e:
                        logger.error(f"Error in main loop processing: {e}")
                        cv2.imshow('Enhanced Live Monitoring - Press Q to quit', frame)
                
                frame_count += 1
                
                # Calculate and display FPS periodically
                if frame_count % 30 == 0:
                    fps = frame_count / (time.time() - start_time)
                    logger.debug(f"Processing FPS: {fps:.2f}")
                
                # Print stats every 50 frames
                if frame_count % 50 == 0:
                    self.print_enhanced_monitoring_stats()
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    # Stop continuous beeping if active
                    if self.continuous_beep_active:
                        self.stop_continuous_beeping()
                    break
                
        except Exception as e:
            logger.error(f"Monitoring error: {e}")
            print(f"❌ Error during monitoring: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_monitoring = False
            self.continuous_beep_active = False
            logger.info("Monitoring stopped")
            print("🛑 Enhanced monitoring stopped")
    
    def print_enhanced_monitoring_stats(self):
        """Print enhanced monitoring statistics"""
        try:
            total_students = len(self.student_data)
            present_students = sum(1 for s in self.student_data.values() if s.get('current_status') == 'present')
            suspicious_students = sum(1 for s in self.student_data.values() if s.get('current_status') == 'suspicious')
            cheating_students = sum(1 for s in self.student_data.values() if s.get('current_status') == 'cheating')
            
            print(f"\n📊 ENHANCED MONITORING STATISTICS:")
            print(f"   Total Students: {total_students}")
            print(f"   Present: {present_students}")
            print(f"   Suspicious: {suspicious_students}")
            print(f"   Cheating: {cheating_students}")
            print(f"   Suspicious Behaviors Detected: {self.suspicious_behavior_count}")
            print(f"   Face Encodings: {len(self.known_face_encodings)}")
            print(f"   Continuous Alert: {'ACTIVE 🚨' if self.continuous_beep_active else 'Inactive'}")
        except Exception as e:
            logger.error(f"Error printing stats: {e}")

    def display_suspicious_behaviors(self):
        """Display all suspicious behaviors that the system can detect"""
        print("\n" + "="*60)
        print("🔍 SUSPICIOUS BEHAVIORS DETECTION CAPABILITIES")
        print("="*60)
        for category, behaviors in suspicious_behaviors.items():
            print(f"\n📂 {category}:")
            for i, behavior in enumerate(behaviors, start=1):
                print(f"   {i}. {behavior}")
        print(f"\nTotal behavior categories: {len(suspicious_behaviors)}")
        print("The system will beep continuously and save photos when high-severity cheating is detected!")
        print("="*60)

    # ==================== MAIN MENU ====================
    
    def main_menu(self):
        """Main menu system"""
        while True:
            print("\n" + "="*50)
            print("🎓 COMPREHENSIVE EXAM HALL SYSTEM")
            print("="*50)
            print(f"📁 Database: {self.db_file}")
            print("📊 REGISTRATION & VERIFICATION:")
            print("1. Register New Student (Fingerprint + Face Camera)")
            print("2. Verify Student for Exam (Fingerprint Only)") 
            print("3. View Registered Students")
            print("4. Add Face Data to Existing Student (Camera)")
            print("5. View Face Data Information")
            print("6. Test Camera")
            
            print("\n🎯 SEAT VERIFICATION:")
            print("7. Enhanced Seat Verification (Face Recognition)")
            
            print("\n🚨 ENHANCED LIVE MONITORING:")
            print("8. Start Enhanced Live Exam Monitoring")
            print("9. View Suspicious Behaviors List")
            print("10. Exit")
            print("-" * 50)
            
            choice = input("Choose option (1-10): ").strip()
            
            if choice == '1':
                self.register_student()
            elif choice == '2':
                self.verify_student()
            elif choice == '3':
                self.view_students()
            elif choice == '4':
                self.add_face_to_existing_student()
            elif choice == '5':
                self.view_face_data_info()
            elif choice == '6':
                self.test_camera()
            elif choice == '7':
                self.run_seat_verification()
            elif choice == '8':
                print("🚀 Starting Enhanced Live Monitoring...")
                print("Press 'q' or ESC to stop monitoring")
                print("🔊 Continuous beeping for high-severity cheating!")
                print("📸 Automatic photo capture for evidence!")
                print("Starting monitoring in 3 seconds...")
                time.sleep(3)
                self.start_live_monitoring()
            elif choice == '9':
                self.display_suspicious_behaviors()
            elif choice == '10':
                print("👋 Thank you for using the Comprehensive Exam Hall System!")
                break
            else:
                print("❌ Invalid choice! Please enter 1-10.")

# Debug function to test components
def test_system_components():
    """Test individual system components before starting monitoring"""
    print("🧪 Testing system components...")
    
    system = ComprehensiveExamSystem()
    
    # Test camera
    print("1. Testing camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print("   ✅ Camera working")
        else:
            print("   ❌ Camera cannot capture frames")
        cap.release()
    else:
        print("   ❌ Camera not accessible")
    
    # Test YOLO model
    print("2. Testing YOLO model...")
    if system.yolo_model is not None:
        print("   ✅ YOLO model loaded")
    else:
        print("   ❌ YOLO model failed to load")
    
    # Test face encodings
    print("3. Testing face encodings...")
    print(f"   ✅ Loaded {len(system.known_face_encodings)} face encodings")
    
    # Test student data
    print("4. Testing student data...")
    print(f"   ✅ Loaded {len(system.student_data)} students")
    
    # Test registered faces
    print("5. Testing registered faces...")
    print(f"   ✅ Loaded {len(system.registered_faces)} registered faces")
    
    # Test beep sound
    print("6. Testing beep sound...")
    try:
        winsound.Beep(500, 200)
        print("   ✅ Beep sound working")
    except:
        print("   ⚠️  Beep sound not available (Windows only)")
    
    print("\n✅ System test completed. Starting main system...")
    return system

# Run the system
if __name__ == "__main__":
    print("🚀 Comprehensive Exam Hall System Starting...")
    
    # Test components first
    system = test_system_components()
    
    # Display suspicious behaviors at startup
    system.display_suspicious_behaviors()
    
    # Start main menu
    system.main_menu()
    
    print("✅ System shutdown complete.")