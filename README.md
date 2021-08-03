# **People-centric Traffic Hotspot Identification and Characterization Based on Social Media Data**

This repository presents the project description and codes for the traffic hotspot analysis. 

## 1. Introduction

In this project, we want to identify and characterize the traffic hotspots based on social media data.

The **research questions** are as follows:

- Could social media data be used to find the traffic hotspots?
  - How to detect the accident hotspots and congestion hotspots based on traffic-related social media?
  - How the accident and congestion hotspots spatially distribute in the city?
  - What is the correlation between the accident hotspot and congestion hotspot detected by social media and actual traffic accident and congestion records?

- Where (route/intersection/expressway/etc) people are most discussed in the accident hotspot and the congestion hotspot? Which hotspot areas should be prioritized for the following traffic management?

The **main contributions** of this study are:

- We use a new data source, social media data, to detect and characterize the accident and congestion hotspots from people’s perspectives. 
- We modify the Kernel Density Estimation (KDE) approach to identify the people-centric traffic hotspots, by incorporating the sentiment information of traffic-relevant microblogs. The detected hotspot is further compared with the traditional KDE method.
- We further provide insights into the traffic accidents and congestions in Shanghai and offer policy recommendations for future traffic management.

## 2. Traffic Hotspot Analysis Module Based on Social Media Data

The overview of the traffic hotspot analysis module is given below:

![Traffic Hotspot Analysis Module](https://github.com/bright1993ff66/traffic_info_perception/blob/main/project_figures/traffic_hotspot_framework.png)

## 3. Prerequisites

Main Python packages for this project are given below:

- GIS Analysis: [geopandas](https://geopandas.org/index.html), [rasterio](https://rasterio.readthedocs.io/en/latest/), [ArcMap 10.4.1](https://desktop.arcgis.com/en/arcmap/10.4/get-started/setup/arcgis-desktop-quick-start-guide.htm) or above
- Traffic Information Detection: [Keras](https://keras.io/)
- Text Processing: [spacy](https://spacy.io/), [gensim](https://radimrehurek.com/gensim/)
- Data management: [pandas](https://pandas.pydata.org/)
- Visualizations: [matplotlib](https://matplotlib.org/), [wordcloud](https://amueller.github.io/word_cloud/)

