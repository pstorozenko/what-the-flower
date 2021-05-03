import streamlit as st
import os
from PIL import Image
from flowermodel import FlowerPowerNet

from torchvision import transforms
import numpy as np
from skimage import io
import torch
from torch.nn import functional as F
from PIL import Image
import pandas as pd

model = FlowerPowerNet.load_from_checkpoint("flower_model.ckpt")
model.eval()
transform = transforms.Compose([
      transforms.Resize(255),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

classes = sorted(['Daffodil','Snowdrop', 'Lily_Valley', 'Bluebell',
           'Crocus', 'Iris', 'Tigerlily', 'Tulip',
           'Fritillary', 'Sunflower', 'Daisy', 'Colts_Foot',
            'Dandelalion', 'Cowslip', 'Buttercup', 'Windflower',
            'Pansy'])
classes = [e.replace("_", " ") for e in classes]

st.title("What the flower 💮💮💮")
st.write("""
How often have seen a flower in the park and wondered what is its name?\n
🌻💮🌺🌼🌸🌹🌷💐\n
Now it's easier then ever before!\n
Just upload flower image and I'll tell you which species it is!
""")
uploaded_file = st.file_uploader("Choose an image with a flower...", type=["jpg","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    tens = transform(image)
    y_hat = model(tens.unsqueeze(0))
    label = classes[y_hat.argmax().item()]

    st.write(f"This is {label}")
    st.image(image, caption=f'{label}', use_column_width=True)
    
    probs = F.softmax(y_hat, dim=1).detach().numpy()
    df = pd.DataFrame({"Species probability" : probs[0,:]}, index=classes)
    st.write("Table of species probability:")
    st.bar_chart(df)


st.write("---------------")
st.write("As for now I recognize the following:")
st.write("* " + "\n* ".join(classes))