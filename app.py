import gradio as gr
from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect(image):
    results = model(image)
    output = results[0].plot()
    return output

interface = gr.Interface(
    fn=detect,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="🚀 Object Detection using YOLOv8",
    description="Upload an image and detect objects."
)

interface.launch()
