# Covid-19 Fake News Detector

## Introduction
In 2022, COVID-19 misinformation has become more widespread than ever through the use of social media, as so many news articles and social media posts are being published every single day that the task of identifying which are real or misinformation has become increasingly difficult. Therefore, there is a need to be able to identify this news as accurately as possible in an automated fashion that is able to keep up with the massive scale of news currently available as it comes in. As a result, this project aims to perform an analysis using a dataset of 10,700 social media posts and articles of both real news & misinformation using various big data technologies.

At a high-level, our project uses several different technologies. We first stored our data in MongoDB, which is a NoSQL database. We fetched the data from the database into a notebook and used PySpark to pre-process the data as well as train and apply a machine learning model to generate predictions as to whether the news was misinformation or not. We also implemented Spark Streaming to be able to provide input texts into the pipeline to make predictions on the spot.

## Demo
![t](demo.gif)
