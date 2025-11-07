import cv2
import os
import yaml
import numpy as np
from pathlib import Path

def load_cfg(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def detect_faces(image_path):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    return faces

def build_gallery(cfg):
    print("[*] Building face gallery...")
    
    gallery_path = cfg.get('gallery', {}).get('root', 'data/gallery')
    if not os.path.exists(gallery_path):
        print(f"Gallery directory {gallery_path} does not exist")
        return
    
    # Create output directories if they don't exist
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image in the gallery
    for person_dir in os.listdir(gallery_path):
        person_path = os.path.join(gallery_path, person_dir)
        if os.path.isdir(person_path):
            print(f"Processing {person_dir}...")
            
            for img_file in os.listdir(person_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(person_path, img_file)
                    faces = detect_faces(img_path)
                    
                    if len(faces) > 0:
                        print(f"  Found {len(faces)} face(s) in {img_file}")
                        # Here you would typically save the face encodings
                        # For now, we'll just draw rectangles around the faces
                        img = cv2.imread(img_path)
                        for (x, y, w, h) in faces:
                            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                        # Save the output
                        output_path = os.path.join(output_dir, f"detected_{img_file}")
                        cv2.imwrite(output_path, img)
                        print(f"  Saved detected faces to {output_path}")
                    else:
                        print(f"  No faces found in {img_file}")

def main():
    cfg = load_cfg()
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-gallery', action='store_true')
    args = parser.parse_args()

    if args.build_gallery:
        build_gallery(cfg)
        return

    # For real-time face detection from webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Press 'q' to quit")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the output
        cv2.imshow('Face Detection', frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse
    main()