# Human-Aligned Hazardous Driving (HAHD) Project

**For more detailed Research Documentation, visit:** [HAHD Research Documentation](https://docs.google.com/document/d/1hRM_BWAT8Vs34GIxIm1OU-KfYSr0TqH3lmXpL6CtJ5c/edit?usp=sharing)

## Overview
The Human-Aligned Hazardous Driving (HAHD) project is an initiative focused on collecting, processing, and analyzing driving behavior data to train machine learning models that align autonomous vehicle decision-making with human driving tendencies. This project consists of three main components:

---
## Folder Structure 
```
HAHD/
├── data/
│   ├── processed/              # Processed data after running the transform and processing (ETL)
│   |       ├── driving_videos/ 
│   |       ├── badgazedata.csv 
│   |       ├── normalized_gaze_data.csv 
│   |       ├── final_user_survey_data.csv
│   |       ├── binned_video_dat_wo_user.csv
│   |       ├── aggregate_gaze_data_by_video.csv
│   ├── raw/
│   |       ├── driving_videos/ # Videos from s3 bucket after running extraction (ETL)
│   |       ├── survey_results_raw.csv # Data from MongoDB running extraction (ETL)
│   |       ├── users_data.csv # Data from MongoDB running extraction (ETL)
├── EDA/                        # EDA Folder
├── ETL/                        # Folder with ETL process
├── frontend/                   # code for frontend of the data collection (survey) web app
├── server/                     # code for backend of the data collection (survey) web app
├── VideoProcessingManagement   # code to process the driving footage before upload to S3 bucket
├── .env                         
├── README.md  
├── package.json 
├── package-lock.json                  
├── .gitignore    
├── requirements.txt    
├── sumulationGazePipeline.py   # TBD                 
```
---

## Documentation Links

- **Video Processing & Data Upload:**  
  [VideoProcessingManagement README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/tree/main/VideoProcessingManagement/README.md)

- **Driving Simulation Web Application:**  
  - [Frontend README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/blob/main/frontend/README.md)  
  - [Backend README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/tree/main/server/README.md)

- **ETL Pipeline:**  
  [ETL README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/tree/main/ETL/README.md)

- **Exploratory Data Analysis:**  
  [EDA README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/tree/main/EDA/README.md)

- **Model Training and Evaluation:**  
  [Models README](https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection/tree/main/models/README.md)

---


## Getting Started

## Step 1: Clone the Repository

```bash
git clone https://github.com/Onyx-AI-LLC/Human-Alignment-Hazardous-Driving-Detection.git
cd Human-Alignment-Hazardous-Driving-Detection
```

## Step 2: Create and Activate a Virtual Environment

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Required Dependencies

```bash 
pip install -r requirements.txt
```

## Step 4: Run the Main Script

```
python main.py
```

--
**This research is made possible due to collaboration between Duke University & Onyx AI LLC.**
