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

## 2. Study Area

We choose Shanghai as our study area, which is one of major cities in China. The overview of Shanghai road network is presented as the following:

![Shanghai Road Network](https://github.com/bright1993ff66/traffic_info_perception/blob/main/project_figures/Shanghai_road_network.png)

## 3. Traffic Hotspot Analysis Module Based on Social Media Data

The overview of the traffic hotspot analysis module is given below:

![Traffic Hotspot Analysis Module](https://github.com/bright1993ff66/traffic_info_perception/blob/main/project_figures/traffic_hotspot_framework.png)

The **main contributions** of this study are:

- We use a new data source, social media data, to detect and characterize the accident and congestion hotspots from people’s perspectives. 
- We modify the Kernel Density Estimation (KDE) approach to identify the people-centric traffic hotspots, by incorporating the sentiment information of traffic-relevant Chinese microblogs. The detected hotspot is further compared with the traditional KDE method.
- We further provide insights into the traffic accidents and congestions in Shanghai and offer policy recommendations for future traffic management.

## 4. Main Findings

> To be updated...

## 5. Descriptions

- The [content analysis](https://github.com/bright1993ff66/traffic_info_perception/tree/main/content_analysis) folder saves the codes to conduct the microblog content analysis, including sentiment analysis, keyword extraction, visualizations, topic modeling ,etc.
- The [detect traffic](https://github.com/bright1993ff66/traffic_info_perception/tree/main/detect_traffic) directory saves the utilities for the traffic-relevant information detection with location information
- The [compute_kde.py](https://github.com/bright1993ff66/traffic_info_perception/blob/main/gis_analysis_project/compute_kde.py) and [compare_kde.py](https://github.com/bright1993ff66/traffic_info_perception/blob/main/gis_analysis_project/compare_kde.py) in the [gis_analysis_project](https://github.com/bright1993ff66/traffic_info_perception/tree/main/gis_analysis_project) directory stores the functions to conduct the [Kernel Density Estimation (KDE)](https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-analyst/how-kernel-density-works.htm#:~:text=Kernel%20Density%20calculates%20the%20density,is%20fitted%20over%20each%20point.&text=The%20density%20at%20each%20output,overlay%20the%20raster%20cell%20center.) and compare the different hotspot identification modules.
- The [text_preprocessing.py](https://github.com/bright1993ff66/traffic_info_perception/blob/main/process_text/text_preprocessing.py) in the [process_text](https://github.com/bright1993ff66/traffic_info_perception/tree/main/process_text) folder saves the functions for text preprocessing.

## 6. Prerequisites

For kernel density computation presented in [compute_kde.py](https://github.com/bright1993ff66/traffic_info_perception/blob/main/gis_analysis_project/compute_kde.py), you need to use the Python version (Python 2.7) installed with ArcMap. For other codes, Python 3.7 or above is needed. 

Main Python packages for this project are given below:

- GIS Analysis: [geopandas](https://geopandas.org/index.html), [rasterio](https://rasterio.readthedocs.io/en/latest/), [ArcMap 10.4.1](https://desktop.arcgis.com/en/arcmap/10.4/get-started/setup/arcgis-desktop-quick-start-guide.htm) or above
- Traffic Information Detection: [Keras](https://keras.io/)
- Text Processing: [spacy](https://spacy.io/), [gensim](https://radimrehurek.com/gensim/)
- Data Management: [pandas](https://pandas.pydata.org/)
- Visualizations: [matplotlib](https://matplotlib.org/), [wordcloud](https://amueller.github.io/word_cloud/)

