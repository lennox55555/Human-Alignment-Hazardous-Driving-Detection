# Human-Aligned Hazardous Driving (HAHD) Project

**For more detailed Research Documentation, visit:** [HAHD Research Documentation](https://docs.google.com/document/d/1hRM_BWAT8Vs34GIxIm1OU-KfYSr0TqH3lmXpL6CtJ5c/edit?usp=sharing)

## Overview
The Human-Aligned Hazardous Driving (HAHD) project is an initiative focused on collecting, processing, and analyzing driving behavior data to train machine learning models that align autonomous vehicle decision-making with human driving tendencies. This project consists of three main components:

1. **Driving Data Collection**
2. **Eye-Tracking Data Collection via a Driving Simulation Web Application**
3. **ETL Pipeline for Machine Learning Models (Coming Soon)**
4. **Training of Deep Learning and ML Models (Coming Soon)**

---
## 0. Folder Structure 
```
HAHD/
├── data/
│   ├── driving_videos/         # Videos from s3 bucket after running ETL/pullVideosAws.py
│   ├── user_gaze_videos/       # Videos with user gaze after running ETL/userGazeVideoCreator.py
│   ├── normalized_gaze_data.csv
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

## 1. Driving Data Collection for the Web Application
The project has accumulated over **20 hours of driving footage** from Durham, North Carolina. Data was collected under **various conditions**, including different times of the day and different weather conditions to ensure diversity in the dataset.

### Camera and Data Specifications
- The driving footage was captured using a **Tesla Model 3 (2021) dashcam**.
- **Number of Cameras**: 4 (Front, Rear, Left Side, Right Side)
- **Primary Recording Camera**: **Front-facing camera**
- **Resolution**: **1280 × 960 pixels**
- **Frame Rate**: **36 FPS (Frames Per Second)**
- **Field of View**: **60°**
- **Video Encoding**: H.264
- **Storage Format**: MP4
- **Bitrate**: **~5 Mbps**
- **Low-Light Performance**: Standard (No night vision enhancements)

### Video Quality
- **Daytime**: Clear and detailed, with high contrast and accurate color representation.
- **Nighttime**: Adequate performance, but may suffer from glare and noise in very low-light conditions.
- **Weather Adaptability**: Works well in rain and fog but may have reduced clarity in extreme conditions.

### Preprocessing Steps
- Videos were **split into 15-second segments** to create manageable training data.
- **Unnecessary camera angles** such as side and rearview dashcams were removed to focus solely on the front-facing driving perspective.
- Implemented **naming conventions** for standardized file management.

---

## 2. Eye-Tracking Data Collection via a Driving Simulation Web Application
A **web application** was developed to allow users to interactively engage with driving scenarios and provide real-time hazard annotations. 

### Features of the Application
1. **Driving Simulation**: Users are presented with a **random driving video** pulled from an AWS S3 bucket.
2. **Hazard Identification**: Users press the **space bar** to mark hazardous instances during the simulation.
3. **Eye-Tracking Data Collection**: The application tracks **eye-gaze data** throughout the video using the **WebGazer.js** machine learning model developed by Brown University.
4. **User Incentives**:
   - Users can receive **referral codes** for participating.
   - Tracks the number of surveys a participant completes to provide incentives.
5. **Authentication & Authorization**:
   - Users must **sign in or create an account** to access the application.
6. **Survey & Instructions Page**:
   - Walks users through the **survey process** and how to use the application correctly.

### Technology Stack
- **Front-end**: React.js
- **Back-end**: Node.js with Express.js
- **Database**: MongoDB

---

## 3. ETL Pipeline for Machine Learning Models (Coming Soon)
- The **ETL (Extract, Transform, Load) pipeline** will handle the structured ingestion of driving footage and user interaction data into a machine learning-ready format.
- Data transformation steps such as **feature extraction**, **cleaning**, and **labeling** will be included.
- Processed data will be **stored in a structured database** for efficient querying and model training.

---

## 4. Training of Deep Learning and ML Models (Coming Soon)
- Future phases will involve training deep learning and traditional ML models on the collected **driving footage and eye-tracking data**.
- Models will focus on **hazard detection, driver attention prediction, and human-aligned decision-making**.
- Techniques such as **computer vision, sequence modeling (LSTMs/Transformers), and reinforcement learning** will be explored.

---

## Contribution & Future Work
- The project is in **active development**, and contributions are welcome.
- Future improvements include **enhanced gaze tracking accuracy**, **additional driving scenarios**, and **more sophisticated hazard prediction models**.

For questions or collaboration opportunities, please reach out!

---

## License
*(Include applicable license information here.)*

---

**This research is made possible due to collaboration between Duke University & Onyx AI LLC.**
