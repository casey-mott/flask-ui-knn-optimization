from flask import Flask, render_template, request, Response, send_file, after_this_request
import pandas as pd
import requests
import json
import config as cfg
import numpy as np
import helper_functions as hlp
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pathlib

application = app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def landing():
    if request.method != 'POST':
        base_path = pathlib.Path('./static/')
        for f_name in base_path.iterdir():
            if str(f_name) != 'static/main.css':
                os.remove(f_name)
        base_path = pathlib.Path('./data/')
        for f_name in base_path.iterdir():
            if str(f_name) != 'data/dummy_data.csv':
                os.remove(f_name)
        return render_template('landing.html')
    else:
        return render_template('landing.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        filename = 'data/full_dataset.csv'
        cleaned_df = hlp.convert_categoricals(df)
        saved = hlp.save_csv_in_memory(cleaned_df, filename)
        columns = []
        for col in cleaned_df.columns:
            columns.append(col)
        return render_template(
            'selections.html',
            columns=columns,
            #sizes=cfg.split_options
        )

    return render_template('home.html')


@app.route('/model_chosen', methods=['GET', 'POST'])
def model_chosen():
    label_pick = request.form.get("label_pick", None)

    label_check, features = hlp.label_test(label_pick, 'data/full_dataset.csv')

    if label_check == 1:
        for split in cfg.split_options[0:4]:

            best, accuracy = hlp.knn(label_pick, split, 'data/full_dataset.csv')

            for file in cfg.file_list[1:]:
                os.remove(file)


        return render_template(
            'knn_single.html',
            line_1='/static/line_65.png',
            line_2='/static/line_70.png',
            line_3='/static/line_75.png',
            line_4='/static/line_80.png'
        )

    else:

        return render_template(
            'model_select.html',
            error=cfg.label_error,
            columns=features
        )


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        df = pd.read_csv(request.files.get('file'))
        filename = 'data/full_dataset.csv'
        cleaned_df = hlp.convert_categoricals(df)
        saved = hlp.save_csv_in_memory(cleaned_df, filename)
        columns = []
        for col in cleaned_df.columns:
            columns.append(col)
        return render_template(
            'model_select.html',
            tests=cfg.tests,
            columns=columns,
        )
    return render_template(
        'home_single.html'
    )


@app.route('/dummy_chosen', methods=['GET', 'POST'])
def dummy_chosen():
    label_pick = request.form.get("label_pick", None)

    label_check, features = hlp.label_test(label_pick, 'data/dummy_data.csv')

    if label_check == 1:
        for split in cfg.split_options[0:4]:

            best, accuracy = hlp.knn(label_pick, split, 'data/dummy_data.csv')

            for file in cfg.file_list[1:]:
                os.remove(file)


        return render_template(
            'knn_single.html',
            line_1='/static/line_65.png',
            line_2='/static/line_70.png',
            line_3='/static/line_75.png',
            line_4='/static/line_80.png'
        )

    else:

        return render_template(
            'model_select.html',
            error=cfg.label_error,
            columns=features
        )


@app.route('/dummy_single', methods=['GET', 'POST'])
def dummy_single():
    dummy_data = pd.read_csv('data/dummy_data.csv')
    columns = dummy_data.columns

    return render_template(
        'dummy_single.html',
        columns=columns
    )


if __name__ == "__main__":
    app.run(host ='0.0.0.0', port = 80, debug = True)
