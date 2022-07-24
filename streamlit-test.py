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
    this_list = [
            ['Task', 'Hours per Day'],
            ['LGBTQ+ Phobic',      0.7],
            ['Sexism',  0.1],
            ['Racial Prejudice', 0.1],
            ['Disability Discrimination',    0.1]
          ]
    st.write(data.head())

#Writes the html/css/javascript: Mostly for the donut chart
components.html(
    """
    <section>
    <div class="donut-chart", style = "position:relative; background-color: transparent;">
  <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
      <script type="text/javascript">
        google.charts.load("current", {packages:["corechart"]});
        google.charts.setOnLoadCallback(drawChart);
        function drawChart() {
          var data = google.visualization.arrayToDataTable(""" + this_list + """);

          var options = {
            title: 'Tone Representation',
            pieHole: 0.4,
            colors: ['#36d8ff', '#529ffc', '#31356e', '#66757f']
          };

          var chart = new google.visualization.PieChart(document.getElementById('donutchart'));
          chart.draw(data, options);
        }
      </script>
      <div id="donutchart" style="width: 700px; height: 350px;"></div></p>
    </div>
    </section>
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