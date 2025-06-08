# About

A simple ML project inspired by the 2nd chapter of [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/), 3rd edition.

This project features a [taxi pricing dataset](https://www.kaggle.com/datasets/denkuznetz/taxi-price-prediction) and techniques from the book to clean the data, train a couple of ML algorithms, choose the best one and deploy it as a pretrained model.

# Structure

`/src/ml` contains all the ML stuff. It downloads the dataset, prepares the data and trains the ML model. It also has some draft code just to serve as a cheat sheet for me in the future. In production, this module should simply expose `get_model` method, to load the pretrained model. This is why almost all the ML libraries are listed as dev dependencies.

`/src/app` contains a simple FastApi server with template page, where user can input some taxi fare data and get the predicted price.

# Demo

https://taxi-price-predictor-b4jv.onrender.com/

<img width="1194" alt="image" src="https://github.com/user-attachments/assets/14e0a5cd-c4a8-486b-94f1-a88a73b42688" />

