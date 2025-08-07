from fastapi import FastAPI
import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image.Image) -> str:
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Define Gradio Interface
gradio_app = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(label="Upload Image", type="pil"),
    outputs=gr.Textbox(label="Generated Caption"),
    title="AI Image Captioning"
).launch(share=True)

# Create FastAPI app
app = FastAPI()

# Mount Gradio inside FastAPI
app = gr.mount_gradio_app(app, gradio_app, path="/")


