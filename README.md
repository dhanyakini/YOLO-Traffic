# Traffic Analysis and Prediction using Flask and YOLOv8

## Overview
This project implements a web-based traffic analysis and prediction system using Flask, YOLOv8, and various other libraries. The application processes live video streams to detect and count vehicles, and uses a pre-trained machine learning model to predict traffic conditions based on these counts.

## Features
- Real-time vehicle detection and counting (cars, buses, trucks) using YOLOv8.
- Integration with YouTube live streams for video input.
- Choropleth map visualization for traffic data.
- Predictive analytics for traffic conditions using a pre-trained machine learning model.

## Technologies Used
- **Flask**: Backend framework for handling web requests.
- **YOLOv8**: For real-time object detection.
- **Plotly**: For interactive map visualizations.
- **scikit-learn**: For data preprocessing and prediction.
- **OpenCV**: For video frame processing.
- **yt-dlp** and **pytube**: For fetching YouTube live stream URLs.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dhanyakini/YOLO-Traffic
   cd YOLO-Traffic
   ```
   
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
   
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Ensure the YOLOv8 model weights are present:
   The YOLOv8 model weights are already included in the `yolov8/` directory of this repository.

5. Ensure the required GeoJSON file is present:
   Ensure the GeoJSON file is in the `data/` directory.

6. Ensure you have the pre-trained machine learning model file:
   Place the model file (`model.joblib`) in the project directory.

7. Train the Random Forest Classifier (Optional):
   The Jupyter Notebook file included in the repository is used to train a Random Forest Classifier. The trained model is saved as `model.joblib` and is utilized by `app_realtime.py` for predictions.

## Usage

1. Run the Flask application:
   ```bash
   python3 app_realtime.py
   ```

2. Open the application in your web browser:
   Navigate to `http://127.0.0.1:5000`.

3. Select a live video stream and view real-time traffic analysis and predictions.

## File Structure
```
.
├── app_realtime.py                                     # Main Flask application
├── data
│   └── map.geojson                                     # GeoJSON file for map visualization
│   └── Traffic.csv                                     # Dataset used to train model
├── model.joblib                                        # Trained ML model
├── requirements.txt                                    # Python dependencies
├── templates
│   └── index.html                                      # HTML templates
├── traffic-prediction-detailed-eda-modeling.ipynb      # Jupyter Notebook used to train model
├── yolov8
    ├── coco.txt                                        # Class labels for YOLO
    ├── tracker.py                                      # Tracks Boundary Boxes across frames
    └── yolov8s.pt                                      # YOLOv8 model weights
```


## Future Enhancements
- Add support for multiple ML models.
- Improve vehicle detection accuracy with advanced YOLO configurations.
- Add more live video stream sources.
- Enhance the UI for better user experience.

## Contributing
Feel free to fork the repository and submit pull requests. Contributions are welcome!

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
