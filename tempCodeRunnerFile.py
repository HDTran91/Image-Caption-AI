import gradio as gr
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image: Image) -> str:
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption

# Define Gradio Interface
image = gr.Image(label="Upload Image")
caption = gr.Textbox(label="Generated Caption")
gradio_interface = gr.Interface(
    fn=generate_caption,
    inputs=image,
    outputs=caption,
    title="AI Image Captioning"
).launch(share=True)
