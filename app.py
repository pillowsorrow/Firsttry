import tkinter as tk
from PIL import ImageTk, Image, ImageDraw, ImageFont
from torch import autocast
import customtkinter as ctk
from authtoken import auth_token
from diffusers import StableDiffusionPipeline
from googletrans import Translator  # Install this library using: pip install googletrans==4.0.0-rc1
# Import torch module
import torch

# Create the app
app = tk.Tk()
app.geometry("532x700")
app.title("Chandrakasem Rajabhat ")
ctk.set_appearance_mode("dark")

image1 = Image.open("Images/Chandrakasem1.png")
photo1 = ImageTk.PhotoImage(image1)
image_label = tk.Label(app, image=photo1)
image_label.place(x=1, y=1)

prompt = ctk.CTkEntry(master=app, height=40, width=370, text_color="black", fg_color="white")
prompt.place(x=140, y=45)

lmain = ctk.CTkLabel( height=512, width=512, master=app )
lmain.place(x=10, y=180)



modelid = "CompVis/stable-diffusion-v1-4"
device = "cuda"  # Use GPU if available
pipe = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipe.to(device)

translator = Translator()

        
def generate():
    try:
        with autocast(device):
            # Get the input text
            thai_text = prompt.get()

            # Translate the text to English
            translation = translator.translate(thai_text, src='th', dest='en')

            # Access the translated text and generate image
            translated_text = translation.text
            print(f"Translated text: {translated_text}")

            # Add logic based on the translated text
            if len(translated_text) > 1 and len(translated_text) < 45:
                # Generate image using translated text
                result = pipe(translated_text, guidance_scale=7)
                generated_image = result.images[0]

                # Draw translated text on the generated image
                draw = ImageDraw.Draw(generated_image)
                draw.text((10, 10), translated_text,  fill="white")

                generated_image.save('generatedimage.png')
                img = ImageTk.PhotoImage(generated_image)
                lmain.configure(image=img)
            else:
                print("ข้อความมีความยาวเกินที่กำหนด")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดตอนกำลังสร้างรูปภาพ: {e}")



trigger = ctk.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="Green", command=generate)
trigger.configure(text="สร้างรูปภาพ")
trigger.place(x=260, y=90)


text1 = tk.Label(app,  text="มหาวิทยาลัยราชภัฏจันทรเกษม สาขาวิทยาการคอมพิวเตอร์", fg="Black" ,font=("Arial",13))
text1.place(x=135, y=12)





app.mainloop()
