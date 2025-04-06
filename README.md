
# Zero-Shot Object Detection with OWL-ViT

This project demonstrates how to perform zero-shot object detection using the OWL-ViT model from Hugging Face's Transformers library. The application captures video input, processes each frame to detect objects based on user-defined text prompts, and displays the annotated video with bounding boxes and labels. It also includes features like TorchScript acceleration and a minimal Tkinter dashboard for real-time detection visualization.


## Setup Instructions

Clone the Repository:

```bash
git clone https://github.com/yourusername/owlvit-object-detection.git
cd owlvit-object-detection
```
Install Required Libraries:
```bash
pip install -r requirements.txt

```
Run the Script:
```bash
python main.py
```
By default, the script uses the webcam as the video source. To use a video file instead:
```bash
python main.py --video_source path/to/video.mp4
```
#### Model download/usage steps
```bash
from transformers import OwlViTProcessor, OwlViTForObjectDetection
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
```
#### Interact with the Application:

-Press 'e' to edit the detection prompts live.

-Press 'q' to quit the application.

## Lessons Learned

#### How It Works
The application utilizes the OWL-ViT model for zero-shot object detection, enabling the detection of objects based on textual descriptions without prior training on those specific classes. It captures video frames, processes them through the model using user-defined prompts, and displays the results in real-time. TorchScript is employed to potentially accelerate inference, and a Tkinter dashboard provides a textual overview of detections.

#### Challenges Faced
Implementing real-time object detection posed challenges such as ensuring smooth video processing while running inference, managing dependencies across different platforms, and integrating a user-friendly interface for prompt editing and detection visualization. Balancing performance and usability required careful consideration.

#### Potential Improvements
Future enhancements could include:

-ONNX Integration: Implementing ONNX for model inference could further accelerate processing times.

-Enhanced UI: Developing a more sophisticated user interface for better interaction and visualization.

-Extended Model Support: Incorporating other models to expand detection capabilities and accuracy.

-These improvements would enhance the application's performance, usability, and versatility in various object detection scenarios.

## Demo Video  
[Watch the Demo Video](https://www.youtube.com/watch?v=8QasvHnEK54)


    
