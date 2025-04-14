# Phishing-website-detection
## 📸 UI Preview
![phishing website detection](https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExeHdsaHQ0dHFicnlpZHFwaTA2d3hkNmgyZGpxbGxiMHVjN29yNWg2NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/f9WFjPrJbcNm2t69Bd/giphy.gif)

## Overview
This project implements a machine learning-based phishing URL detection system using XGBoost with feature optimization. The system analyzes URL characteristics to identify potential phishing attempts with high accuracy.

## Features
-    Real-time URL analysis: Check individual URLs instantly

-    Batch processing: Analyze multiple URLs from a CSV file

-    Feature extraction: 115+ URL features extracted using NLP techniques

-    XGBoost model: Optimized classifier with swarm intelligence feature selection

-    User-friendly interface: Streamlit web application

-    Model explainability: Shows key features influencing each decision

## Dataset
The model was trained on a dataset containing the following features:

```bash 
['qty_dot_url', 'qty_hyphen_url', 'qty_underline_url', 'qty_slash_url', 
'qty_questionmark_url', 'qty_equal_url', 'qty_at_url', 'qty_and_url', 
... (115+ features) ...
'url_shortened', 'phishing']
```
## Installation
Clone the repository:

```bash
git clone https://github.com/yourusername/phishing-website-detection.git
cd phishing-website-detection
```
Create and activate a virtual environment:

``` bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
Install dependencies:

```bash
pip install -r requirements.txt
```
Usage
First train the model (requires your dataset):

```bash

python train_model.py
```
Run the web application:

```bash

streamlit run app.py
```
Access the application in your browser at http://localhost:8501

## Application Components
-    Single URL Analysis: Enter a URL to check if it's phishing

-    Batch Analysis: Upload a CSV file with multiple URLs

-    Model Info: View information about the model and methodology

## Project Structure

phishing-url-detector/
  ├── app.py                # Streamlit web application
  ├── train_model.py        # Model training script
  ├── requirements.txt      # Python dependencies
  ├── phishing_model.joblib # Trained model (generated after training)
  ├── README.md             # This file
  └── data/                 # Dataset directory (not included in repo)
## Methodology
-    URL Collection: Gather URLs from various sources

-    Feature Extraction: NLP-based techniques to extract URL characteristics

-    Feature Optimization: Binary Bat Algorithm (SI-BBA) for feature selection

-    Model Training: XGBoost classifier with optimized features

-    Real-Time Detection: Web interface for instant analysis

-    Feedback Loop: Continuous model improvement

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
PhishTank and OpenPhish for phishing datasets

XGBoost developers

Streamlit team for the web framework
