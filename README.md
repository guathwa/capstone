## DSI-10 Capstone Project (Automatic Tex Summarization)

## A Generic Text Summarizer

## docker image : https://hub.docker.com/repository/docker/guathwa/docker-flask-summary

## telegram bot :  https://web.telegram.org/#/im?p=@TextSummarizerBot

## flask web prototype : https://guathwa.wixsite.com/portfolio/projects

## Problem Statement
Having problem with too much content to read but too little time ? Get a summary that contains the key sentences using an extractive text summarizer.

## Executive Summary
Automatic Text Summarization is one of the most challenging and interesting problems in the field of Natural Language Processing (NLP). It is a process of generating a concise and meaningful summary of text from multiple text resources such as books, news articles, blog posts, research papers, emails, and tweets.

The demand for automatic text summarization systems is spiking these days thanks to the availability of large amounts of textual data.

In this project, we will work on single-document text summarization which is the task of automatically generating a shorter version of a document while retaining its most important information.

This summarizer is able to read in a document from a text file, a PDF, from a URL or a block of text, uses three different models (Graph Based, Centriod Based, Pre-Trained Bert) to generate a summary based on the settings (by number of sentences, 1 min, 3 mins, 5 mins summary).

## Conclusions
Based on model evaluation results on BBC dataset and user scoring, all 3 models appear to generate summaries reasonably well, with model 2 (centroid based BOW summarization) being the best in terms of speed and quality of summary.

A prototype was built using flash and deployed to AWS EC2 instance. This prototype includes all the functionalities document in this notebook except model 3 - Pretrained bert summarization as it was not able to run with 1gb ram provided in the EC2 free-tier.

This project can be further expanded to include multiple document domains, abstractive summarization and a mobile version that can be deployed on mobile platform to enhance accessibility.
