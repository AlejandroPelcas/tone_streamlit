import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests

header = st.container()
dataset = st.container()
models = st.container()

with header:
    st.title("Welcome to our Capstone Project!")

with dataset:
    st.header("We are team Tone. ")
    st.text("Our  dataset is composed of 24,000 tweets. Take a look")
    data = pd.read_csv("/Users/alejandropelcastre/Documents/capstone/data/text_emotion.csv", encoding = "ISO-8859-1")

    st.write(data.head())
########
    sentence = st.text_input('Input your sentence here:')

if sentence:
    st.write(my_model.predict(sentence))

########

    #train models
    from sklearn.model_selection import train_test_split

    X = data.content
    y = data.sentiment

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Creating a donut chart of the proportion of each emotion (to provide a representation of the end product):
    df = data
    names = list(df['sentiment'].value_counts().index)
    size = list(df['sentiment'].value_counts().values)
    size = [i/sum(list(df['sentiment'].value_counts().values)) for i in size]

    my_circle = plt.Circle( (0,0), 0.7, color='white')
    plt.pie(size, labels=names, colors=['aqua', 'cadetblue', 'royalblue', 'blue',  'cornflowerblue', 'cyan', 'darkslategray', 'deepskyblue', 'dodgerblue', 'lightblue', 'oldlace', 'royalblue'])
    p = plt.gcf()
    plt.tight_layout()
    p.gca().add_artist(my_circle)
    plt.tight_layout()

    plt.title('Donut Chart of Sentiment Proportions')
    plt.show()
    st.pyplot(p)

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
