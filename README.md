# Telecommunication Churn Prediction

## Project Overview

This project predicts customer churn for a telecommunication company using machine learning. It provides a web interface for both single and batch predictions, visualizes feature importances, and allows users to upload their own data for analysis.

## Features

- Predicts if a customer is likely to churn based on their attributes
- Batch prediction via CSV upload
- Visualizes top features influencing churn
- Clean, modern web interface using Flask and Bootstrap
- Uses a pre-trained Random Forest model

## Directory Structure

```
Application/
├── app.py                  # Main Flask application
├── model.sav               # Pre-trained churn prediction model
├── first_telc.csv          # Sample data for feature engineering
├── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Main dataset
├── test_data1.csv          # Output of batch predictions
├── result_data.csv         # Feature importance and probability results
├── templates/
│   ├── home.html           # Main prediction form
│   ├── index.html          # Batch upload form
│   ├── results.html        # Batch prediction results
│   └── style.css           # Custom styles
├── static/
│   └── upload.png          # Logo for upload page
```

## Setup Instructions

1. Clone the repository and navigate to the project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   cd Application
   python app.py
   ```
4. Open your browser and go to `http://localhost:5000/`.

## Usage

- **Single Prediction:** Fill out the form on the home page and submit to get churn prediction and feature importance.
- **Batch Prediction:** Go to the Upload page, select a CSV file with customer data, and upload. Results will be displayed in a table and saved as `test_data1.csv`.

## Data Description

- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: Main dataset with customer features and churn labels.
- `first_telc.csv`: Sample data for feature engineering and form structure.
- `test_data1.csv`: Output of batch predictions (churned customers and key features).
- `result_data.csv`: Feature importance and churn probability for each customer.

## Model

- The model is a pre-trained Random Forest classifier saved as `model.sav`.
- Features include demographic, account, and service-related attributes.

## Web Interface

- Built with Flask, Bootstrap, and custom CSS.
- Pages:
  - **Home:** Single customer prediction form
  - **Upload:** Batch prediction via CSV upload
  - **Results:** Table of batch prediction results

## Credits

- Data source: [IBM Sample Data Sets](https://www.ibm.com/communities/analytics/watson-analytics-blog/guide-to-sample-datasets/)

---

For any issues or contributions, please open an issue or pull request.
