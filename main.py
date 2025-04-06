import cv2
import time
import csv
import argparse
import threading
from PIL import Image
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection

# For minimal dashboard UI
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

# Global variable to store latest detections for dashboard display
latest_detections = ""


# Dashboard UI Functionality

def start_dashboard():
    """Starts a minimal Tkinter dashboard to display current detections."""
    global latest_detections
    dashboard = tk.Tk()
    dashboard.title("Detection Dashboard")
    text_area = ScrolledText(dashboard, width=50, height=10)
    text_area.pack(padx=10, pady=10)

    def update_text():
        text_area.delete('1.0', tk.END)
        text_area.insert(tk.END, latest_detections)
        dashboard.after(500, update_text)  # Update every 500ms

    update_text()
    dashboard.mainloop()


# Helper functions
def update_prompts():
    """
    Prompts user to enter new custom detection classes (comma separated).
    Returns a list of strings.
    """
    new_prompts = input("Enter new object classes (comma separated): ")
    prompt_list = [p.strip() for p in new_prompts.split(",") if p.strip()]
    if not prompt_list:
        print("No valid prompt entered. Keeping previous prompts.")
    else:
        print("Updated detection prompts:", prompt_list)
    return prompt_list if prompt_list else None

def draw_detections(frame, detections):
    """
    Draw bounding boxes, labels and scores on the frame.
    Each detection in detections should have 'box', 'score', and 'label' keys.
    """
    for det in detections:
        box = det["box"]  # (x_min, y_min, x_max, y_max)
        score = det["score"]
        label = det["label"]
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (int(box[0]), int(box[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def log_detections(csv_writer, frame_idx, detections):
    """Log detection results to CSV."""
    for det in detections:
        row = [frame_idx, det["label"], det["score"], det["box"]]
        csv_writer.writerow(row)


# Main processing function

def main(args):
    global latest_detections
    # Initialize video capture
    cap = cv2.VideoCapture(args.video_source)
    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # Load the zero-shot object detection model and processor (OWL-ViT)
    print("Loading OWL-ViT model...")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Attempt TorchScript acceleration
    try:
        scripted_model = torch.jit.script(model)
        model = scripted_model
        print("Model successfully scripted with TorchScript.")
    except Exception as e:
        print("TorchScript scripting failed, using original model. Error:", e)

    # Set initial custom detection prompts (non-COCO classes)
    detection_prompts = [
        "a lightbulb", 
        "a matchstick", 
        "a monitor", 
        "a lion", 
        "a computer mouse"
    ]
    print("Using detection prompts:", detection_prompts)

    # Start the dashboard UI in a separate thread
    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    # Prepare CSV logging
    csv_file = open("detections_log.csv", mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame_index", "label", "score", "box"])  # header

    # Frame processing loop
    frame_idx = 0
    fps_start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames or cannot read the video.")
            break
        
        orig_frame = frame.copy()
        frame_idx += 1

        # Convert frame (BGR) to PIL Image (RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Prepare inputs for the model with custom prompts
        inputs = processor(images=pil_image, text=detection_prompts, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Post-process outputs to get bounding boxes and scores
        target_sizes = torch.tensor([pil_image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_sizes)[0]

        # Format detections for drawing, logging, and dashboard display
        detections = []
        for score, label, box in zip(results["scores"].cpu().numpy(), 
                                     results["labels"], 
                                     results["boxes"].cpu().numpy()):
            label_text = detection_prompts[label]
            detection = {
                "score": score,
                "label": label_text,
                "box": box.tolist()
            }
            detections.append(detection)
        
        # Log detections to CSV file
        log_detections(csv_writer, frame_idx, detections)

        # Update dashboard global variable with current detections
        latest_detections = "\n".join([f"{det['label']}: {det['score']:.2f}" for det in detections])

        # Draw detections on the frame
        annotated_frame = draw_detections(orig_frame, detections)
        
        # Calculate and display FPS
        fps = frame_idx / (time.time() - fps_start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10,30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the annotated frame
        cv2.imshow("Zero-shot Object Detection", annotated_frame)
        
        # Handle key presses:
        key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if key == ord('q'):
            break
        # Press 'e' to edit the custom prompts live
        elif key == ord('e'):
            new_prompts = update_prompts()
            if new_prompts is not None:
                detection_prompts = new_prompts

    # Cleanup: release video capture and close CSV file
    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print("Processing complete. Detections logged to detections_log.csv")


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot object detection with OWL-ViT and acceleration/dashboard features")
    parser.add_argument("--video_source", type=str, default="0",
                        help="Path to video file or '0' for webcam. (default: 0)")
    args = parser.parse_args()
    
    # If video_source is '0', convert it to int for webcam capture
    if args.video_source == "0":
        args.video_source = 0

    main(args)
