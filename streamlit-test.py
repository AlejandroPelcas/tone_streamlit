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
    sentence = st.text_input('Input your sentence here:')
######## try html/java
components.html(
    """
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    <div id="accordion">
      <div class="card">
        <div class="card-header" id="headingOne">
          <h5 class="mb-0">
            <button class="btn btn-link" data-toggle="collapse" data-target="#collapseOne" aria-expanded="true" aria-controls="collapseOne">
            Collapsible Group Item #1
            </button>
          </h5>
        </div>
        <div id="collapseOne" class="collapse show" aria-labelledby="headingOne" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #1 content
          </div>
        </div>
      </div>
      <div class="card">
        <div class="card-header" id="headingTwo">
          <h5 class="mb-0">
            <button class="btn btn-link collapsed" data-toggle="collapse" data-target="#collapseTwo" aria-expanded="false" aria-controls="collapseTwo">
            Collapsible Group Item #2
            </button>
          </h5>
        </div>
        <div id="collapseTwo" class="collapse" aria-labelledby="headingTwo" data-parent="#accordion">
          <div class="card-body">
            Collapsible Group Item #2 content
          </div>
        </div>
      </div>
    </div>
    """,
    height=600,
)

    #get animations
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
