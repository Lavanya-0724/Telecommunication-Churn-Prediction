from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.sav", "rb"))
df_1 = pd.read_csv("first_telc.csv")
q = ""


# Define a route for the home page
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/upload")
def index():
    return render_template("index.html")


# Define a route for the prediction form
@app.route("/", methods=["POST"])
def predictForm():
    """
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    """

    inputQuery1 = request.form["query1"]
    inputQuery2 = request.form["query2"]
    inputQuery3 = request.form["query3"]
    inputQuery4 = request.form["query4"]
    inputQuery5 = request.form["query5"]
    inputQuery6 = request.form["query6"]
    inputQuery7 = request.form["query7"]
    inputQuery8 = request.form["query8"]
    inputQuery9 = request.form["query9"]
    inputQuery10 = request.form["query10"]
    inputQuery11 = request.form["query11"]
    inputQuery12 = request.form["query12"]
    inputQuery13 = request.form["query13"]
    inputQuery14 = request.form["query14"]
    inputQuery15 = request.form["query15"]
    inputQuery16 = request.form["query16"]
    inputQuery17 = request.form["query17"]
    inputQuery18 = request.form["query18"]
    inputQuery19 = request.form["query19"]

    model = pickle.load(open("model.sav", "rb"))

    data = [
        [
            inputQuery1,
            inputQuery2,
            inputQuery3,
            inputQuery4,
            inputQuery5,
            inputQuery6,
            inputQuery7,
            inputQuery8,
            inputQuery9,
            inputQuery10,
            inputQuery11,
            inputQuery12,
            inputQuery13,
            inputQuery14,
            inputQuery15,
            inputQuery16,
            inputQuery17,
            inputQuery18,
            inputQuery19,
        ]
    ]

    new_df = pd.DataFrame(
        data,
        columns=[
            "SeniorCitizen",
            "MonthlyCharges",
            "TotalCharges",
            "gender",
            "Partner",
            "Dependents",
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
            "Contract",
            "PaperlessBilling",
            "PaymentMethod",
            "tenure",
        ],
    )

    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

    df_2["tenure_group"] = pd.cut(
        df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels
    )
    # drop column customerID and tenure
    df_2.drop(columns=["tenure"], axis=1, inplace=True)

    new_df__dummies = pd.get_dummies(
        df_2[
            [
                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
                "tenure_group",
            ]
        ]
    )

    # final_df=pd.concat([new_df__dummies, new_dummy], axis=1)

    single = model.predict(new_df__dummies.tail(1))
    probablity = model.predict_proba(new_df__dummies.tail(1))[:, 1]

    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity * 100)

    feature_importances = model.feature_importances_
    feature_names = new_df__dummies.columns

    # Create a DataFrame to display the feature importances
    feature_importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    feature_importance_df = feature_importance_df.sort_values(
        by="Importance", ascending=False
    )

    # Select the top 7 features
    top_features = feature_importance_df.head(7)

    # Define explanations for each feature
    explanations = {
        "Contract_One year": "The customer has a contract duration of one year.",
        "PaperlessBilling_No": "The customer prefers to receive paper bills.",
        "gender_Male": "The customer is male.",
        "tenure_group_1 - 12": "The customer's tenure falls between 1 and 12 months.",
        "InternetService_No": "The customer does not have internet service.",
        "tenure_group_61 - 72": "The customer's tenure falls between 61 and 72 months.",
        "gender_Female": "The customer is female.",
        "DeviceProtection_Yes": "The customer has opted for device protection.",
        "Contract_Two year": "The customer has a contract duration of two years.",
        "StreamingMovies_Yes": "The customer streams movies.",
        "OnlineBackup_No": "The customer does not have online backup.",
        "OnlineSecurity_No internet service": "The customer does not have internet service for online security.",
        "StreamingTV_Yes": "The customer streams TV shows.",
        "TechSupport_No internet service": "The customer does not have internet service for tech support.",
        "InternetService_Fiber optic": "The customer has fiber optic internet service.",
        "OnlineBackup_Yes": "The customer has online backup.",
        "StreamingTV_No": "The customer does not stream TV shows.",
        "TechSupport_Yes": "The customer has opted for tech support.",
        "OnlineSecurity_Yes": "The customer has online security.",
        "PhoneService_No": "The customer does not have phone service.",
        "DeviceProtection_No": "The customer does not have device protection.",
        "PaymentMethod_Electronic check": "The customer prefers to pay using electronic check.",
        "tenure_group_49 - 60": "The customer's tenure falls between 49 and 60 months.",
        "OnlineSecurity_No": "The customer does not have online security.",
        "PaperlessBilling_Yes": "The customer prefers paperless billing.",
        "Dependents_No": "The customer does not have any dependents.",
        "TechSupport_No": "The customer does not have tech support.",
        "OnlineBackup_No internet service": "The customer does not have internet service for online backup.",
        "PaymentMethod_Credit card (automatic)": "The customer prefers to pay using a credit card (automatic).",
        "StreamingTV_No internet service": "The customer does not have internet service for streaming TV shows.",
        "StreamingMovies_No internet service": "The customer does not have internet service for streaming movies.",
        "tenure_group_25 - 36": "The customer's tenure falls between 25 and 36 months.",
        "MultipleLines_Yes": "The customer has multiple phone lines.",
        "tenure_group_13 - 24": "The customer's tenure falls between 13 and 24 months.",
        "tenure_group_37 - 48": "The customer's tenure falls between 37 and 48 months.",
        "Partner_Yes": "The customer has a partner.",
        "MultipleLines_No phone service": "The customer does not have a phone service.",
        "PhoneService_Yes": "The customer has phone service.",
        "gender_Yes": "The customer's gender is unknown.",
        "Partner_No": "The customer does not have a partner.",
        "InternetService_DSL": "The customer has DSL internet service.",
        "StreamingMovies_No": "The customer does not stream movies.",
        "Contract_Month-to-month": "The customer has a month-to-month contract.",
        "PaymentMethod_Electronic mail": "The customer prefers to pay using electronic mail.",
        "PaymentMethod_Bank transfer (automatic)": "The customer prefers to pay using a bank transfer (automatic).",
        "PaymentMethod_Mailed check": "The customer prefers to pay using a mailed check.",
        "DeviceProtection_No internet service": "The customer does not have internet service for device protection.",
        "MultipleLines_No": "The customer does not have multiple phone lines.",
        "Dependents_Yes": "The customer has dependents.",
        "SeniorCitizen_1": "The customer is a senior citizen.",
        "SeniorCitizen_0": "The customer is not a senior citizen.",
    }

    # Initialize the string variable
    o3 = "Top 7 Features Affecting Churn Prediction:\n"

    # Iterate over the top 7 features
    for index, row in top_features.iterrows():
        feature = row["Feature"]
        importance = row["Importance"]
        percentage = importance * 100
        explanation = explanations.get(feature, "Explanation not available")
        o3 += "{}  : {:.2f}%  -  {}\n".format(feature, percentage, explanation)

    # Select the top 7 features
    top_features = feature_importance_df.head(7)

    # Plot the feature importances
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.bar(top_features["Feature"], top_features["Importance"], color="black")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("Top 7 Features Affecting Churn Prediction")
    plt.xticks(rotation=45)

    # Adjust figure margins to avoid cropping
    plt.subplots_adjust(bottom=0.3)

    # Save the plot to a BytesIO object
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    # Encode the plot image as base64
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    buffer.close()

    return render_template(
        "home.html",
        output1=o1,
        output2=o2,
        output3=o3,
        image_base64=image_base64,
        query1=request.form["query1"],
        query2=request.form["query2"],
        query3=request.form["query3"],
        query4=request.form["query4"],
        query5=request.form["query5"],
        query6=request.form["query6"],
        query7=request.form["query7"],
        query8=request.form["query8"],
        query9=request.form["query9"],
        query10=request.form["query10"],
        query11=request.form["query11"],
        query12=request.form["query12"],
        query13=request.form["query13"],
        query14=request.form["query14"],
        query15=request.form["query15"],
        query16=request.form["query16"],
        query17=request.form["query17"],
        query18=request.form["query18"],
        query19=request.form["query19"],
    )


@app.route("/upload", methods=["POST"])
def predictUpload():
    # Get the uploaded Excel file from the form data
    excel_file = request.files["excel_file"]
    # Read the Excel file into a DataFrame
    telco_base_data = pd.read_csv(excel_file)
    telco_data = telco_base_data.copy()
    telco_data.TotalCharges = pd.to_numeric(telco_data.TotalCharges, errors="coerce")
    telco_data.dropna(how="any", inplace=True)
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    telco_data["tenure_group"] = pd.cut(
        telco_data.tenure, range(1, 80, 12), right=False, labels=labels
    )
    telco_customer_id = pd.DataFrame(
        telco_data[["customerID", "SeniorCitizen", "MonthlyCharges", "TotalCharges"]]
    )
    telco_data.drop(columns=["customerID", "tenure"], axis=1, inplace=True)
    telco_data_dummies = pd.get_dummies(telco_data)
    telco_data_dummies = telco_data_dummies[
        [
            "SeniorCitizen",
            "MonthlyCharges",
            "TotalCharges",
            "gender_Female",
            "gender_Male",
            "Partner_No",
            "Partner_Yes",
            "Dependents_No",
            "Dependents_Yes",
            "PhoneService_No",
            "PhoneService_Yes",
            "MultipleLines_No",
            "MultipleLines_No phone service",
            "MultipleLines_Yes",
            "InternetService_DSL",
            "InternetService_Fiber optic",
            "InternetService_No",
            "OnlineSecurity_No",
            "OnlineSecurity_No internet service",
            "OnlineSecurity_Yes",
            "OnlineBackup_No",
            "OnlineBackup_No internet service",
            "OnlineBackup_Yes",
            "DeviceProtection_No",
            "DeviceProtection_No internet service",
            "DeviceProtection_Yes",
            "TechSupport_No",
            "TechSupport_No internet service",
            "TechSupport_Yes",
            "StreamingTV_No",
            "StreamingTV_No internet service",
            "StreamingTV_Yes",
            "StreamingMovies_No",
            "StreamingMovies_No internet service",
            "StreamingMovies_Yes",
            "Contract_Month-to-month",
            "Contract_One year",
            "Contract_Two year",
            "PaperlessBilling_No",
            "PaperlessBilling_Yes",
            "PaymentMethod_Bank transfer (automatic)",
            "PaymentMethod_Credit card (automatic)",
            "PaymentMethod_Electronic check",
            "PaymentMethod_Mailed check",
            "tenure_group_1 - 12",
            "tenure_group_13 - 24",
            "tenure_group_25 - 36",
            "tenure_group_37 - 48",
            "tenure_group_49 - 60",
            "tenure_group_61 - 72",
        ]
    ]

    # single = model.predict(telco_data_dummies.tail(1))
    # probablity = model.predict_proba(telco_data_dummies.tail(1))[:,1]
    # Make predictions using the loaded model
    predictions = model.predict(telco_data_dummies)
    telco_data_dummies["predictions"] = predictions

    # Filter the DataFrame to include only rows with prediction values equal to 1
    filtered_df = telco_data_dummies[telco_data_dummies["predictions"] == 1]
    filtered_df = pd.merge(
        filtered_df,
        telco_customer_id,
        on=["SeniorCitizen", "MonthlyCharges", "TotalCharges"],
    )
    # Extract the desired columns from filtered_df
    columns = [
        "customerID",
        "PaymentMethod_Electronic check",
        "OnlineSecurity_No",
        "TechSupport_No",
        "SeniorCitizen",
        "Contract_Month-to-month",
        "MonthlyCharges",
        "TotalCharges",
        "tenure_group_1 - 12",
        "tenure_group_13 - 24",
    ]
    customer_df = filtered_df[columns].copy()

    # Save the filtered DataFrame to "test_data1.csv"
    customer_df.to_csv("test_data1.csv", index=False)

    # Convert the DataFrame to an HTML table
    result_table = customer_df.to_html(index=False)

    # Return the prediction results and the HTML table
    return render_template(
        "results.html", predictions=predictions, result_table=result_table
    )
    return render_template("results.html")


if __name__ == "__main__":
    app.run(debug=True)
