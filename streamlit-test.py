import streamlit as st
import pandas as pd
import numpy as np
#import os
import matplotlib.pyplot as plt
from streamlit_lottie import st_lottie
import json
import requests
from PIL import Image
import streamlit as st
import streamlit.components.v1 as components
import subprocess
import sys
from pip._internal import main as pipmain

pipmain(['install', "torch"])
pipmain(['install', "torchmetrics"])
pipmain(['install', "pytorch_lightning"])
pipmain(['install', "pylab"])
pipmain(['install', "transformers"])

# from model import *
#######3#model.py##########
# Pip Installing Dependencies
# !pip install torch -q
# !pip install watermark -q
# !pip install transformers -q
# !pip install --upgrade pytorch-lightning -q
# !pip install colored -q
# !pip install -U -q PyDrive -q

# Import Packages
# import pandas as pd
# import numpy as np
# from tqdm.auto import tqdm
import torch
import torchmetrics
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, auroc
from torchmetrics.functional import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, multilabel_confusion_matrix
# import seaborn as sns
from pylab import rcParams
# from matplotlib import rc

# %matplotlib inline
# %config InlineBackend.figure_format='retina'
# RANDOM_SEED = 42
# sns.set(style='whitegrid', palette='muted', font_scale=1.2)
# HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
# sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
# rcParams['figure.figsize'] = 12, 8
# pl.seed_everything(RANDOM_SEED)

class TweetTagger(pl.LightningModule):
  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained('bert-base-cased', return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()
  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output
  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}
  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss
  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss
  def training_epoch_end(self, outputs):
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)
    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)
    for i, name in enumerate(6):
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )
    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
    )

def get_model_predictions(tweet):
    model = TweetTagger(n_classes=6, n_warmup_steps=140, n_training_steps=703)
    loaded_model = TweetTagger(n_classes=6,
                           n_warmup_steps=140,
                           n_training_steps=703)
    
    #cwd = os.getcwd() # getting current working directory
    #print('This is the Current Directory: ')
    #print(cwd + '/pytorch_model.pth')
    loaded_model.load_state_dict(torch.load('./pytorch_model.pth'))
    loaded_model.eval()

    BERT_MODEL_NAME = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

    encoding = tokenizer.encode_plus(tweet, add_special_tokens=True, max_length=512,
    return_token_type_ids=False, padding="max_length", return_attention_mask=True,
    return_tensors='pt',)

    _, test_prediction = loaded_model(encoding["input_ids"], encoding["attention_mask"])
    test_prediction = test_prediction.flatten().detach()
    LABEL_COLUMNS = ['Neutral', 'General Criticsm', 'Disability Shaming', 'Racial Prejudice',
                 'Sexism','LGBTQ+ Phobia']

    result = []
    for label, prediction in zip(LABEL_COLUMNS, test_prediction):
        result.append([label, prediction])

    return result


###########################



header = st.container()
mission = st.container()
dataset = st.container()
models = st.container()

with header:
    #Insert  Title
    st.title("Welcome to our Capstone Project!")
    image = Image.open('tone_log.png')
    st.image(image, caption = "Toning down the bad vibes")

with mission:
    st.title("Mission Statement:")
    st.text("Promoting empathy among Twitter Users to reduce offensive content that harms the wellness of users")

with dataset:
    sentence = st.text_input('Input your sentence here:')
    if sentence:
        answer = get_model_predictions(sentence)
        st.write(answer)
    else:
        answer = [['Neutral', 1.0], ['General Criticism', 0],
        ['Disability Shaming', 0], ['Sexism', 0],
        ['Racial Prejudice', 0], ['LGBTQ+ Phobia', 0]
        ]
    st.text("""The data is composed of 24,000 tweets from the kaggle dataset, Hate Speech and Offensive Language Dataset. It was conceived to be used to research hate speech such
    as racial, homophobic, sexist, and general offensive language. The origional dataset origionally had
    the following columns: hate_speech, offensive_language, and neither. We added more columns. Here, take a look""")
    data = pd.read_csv("multi_label_new.csv", encoding = "ISO-8859-1")
    answer.insert(0, ['Task', 'Hours per Day'])


    st.write(data.tail(10))
    # pred = model.get_model_predictions("I hate james a lot")
    # st.text(pred)

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
          var data = google.visualization.arrayToDataTable(""" + str(answer) + """);

          var options = {
            title: 'Tone Representation',
            pieHole: 0.4,
            colors: ['#36d8ff', '#529ffc', '#31356e', '#66757f', '#5F9EA0', '#96DED1']
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