import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components

header = st.container()
dataset = st.container()
models = st.container()

with header:
    #Insert  Title
    st.title("Welcome to our Capstone Project!")
    image = Image.open('tone_log.png')
    st.image(image, caption = "Toning down the bad vibes")

with dataset:
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        st.write(my_model.predict(sentence))
    st.header("Our Data:")
    st.text("Our dataset is composed of 24,000 tweets. Here, take a look")
    data = pd.read_csv("multi_label_new.csv", encoding = "ISO-8859-1")
    st.write(data.head())
    sentence = st.text_input('Input your sentence here:', key = 7)
######## try html/java
components.html(
    """
    <body>
    <link rel="stylesheet" href="style.css">
    <script type="text/javascript" src="jquery-3.3.1.js"></script>
    <div class="wrapper">
    <div class="pie-charts">
        <div class="pieID--operations pie-chart--wrapper">
        <p align="left"><h2>Tone Representation</h2>
        <div class="pie-chart">
            <div class="pie-chart__pie"></div>
            <ul class="pie-chart__legend">
            <li><em>LGBTQ+ Phobic</em><span>0.50</span></li>
            <li><em>Sexism</em><span>0.30</span></li>
            <li><em>Racial Prejudice</em><span>0.10</span></li>
            <li><em>Disability discrimination</em><span>0.10</span></li>
            </ul>
        </div>
        </div>
        </div>
        </div>

    <script src='https://code.jquery.com/jquery-2.2.4.min.js'></script>
    <script src='https://codepen.io/MaciejCaputa/pen/EmMooZ.js'></script><script  src="./chart.js"></script>
    </body>
    """,
    height=600,
)

""" #get animations
    def load_lottiefile(filepath: str):
        with open(filepath, "r") as f:
            return json.load(f)

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()

    lottie_chart = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_nedzkmew.json")

    st_lottie(
    lottie_chart,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    height=300,
    width=300,
    key=None,
    )
"""