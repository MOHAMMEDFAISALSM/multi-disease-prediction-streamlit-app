Multi-Disease Prediction App
Project Summary
The Multi-Disease Prediction App is an interactive web application built using Python and the Streamlit framework. Its main goal is to assist users—patients, healthcare professionals, and enthusiasts—in predicting their risk for multiple diseases with a single interface. The app leverages machine learning models trained on real medical data to provide risk assessments for:

Diabetes

Heart Disease

Parkinson's Disease

Cardio Disease

Hepatitis

Liver Disease

Ocular Disease

Thyroid Disease

Multiple Sclerosis

Nine different datasets, sourced from Kaggle, power the predictions. Users simply enter relevant medical details, and the app delivers an immediate, data-driven prediction for the selected condition.

Features
Single Platform: Predicts nine major diseases from one unified app.

Easy-to-Use Interface: Built with Streamlit for responsiveness and clarity.

Machine-Learning Powered: Models trained on reputed Kaggle datasets.

Text-to-Speech Integration: gTTS gives users an audio summary of results.

Beautiful Visuals: Data visualization with Matplotlib for insights.

Steps to Run the Application
Clone or Download the Repository

bash
git clone https://github.com/MOHAMMEDFAISALSM/multi-disease-prediction-streamlit-app.git
cd multi-disease-prediction-streamlit-app
Install Required Libraries

Make sure you have Python 3.7+ installed. Run the following command to install the required packages:

bash
pip install -r requirements.txt
If requirements.txt is not available, install individually:

bash
pip install streamlit pandas matplotlib gtts streamlit-option-menu
Download the Datasets

The app uses nine datasets, all of which can be downloaded directly from Kaggle. Refer to the app’s documentation or the Kaggle links for each disease.

Run the Streamlit App

From the project root directory, execute:

bash
streamlit run multi_disease_pred.py
Example:

If your script is named multi_disease_pred.py, run:

bash
streamlit run multi_disease_pred.py
Interact with the Web App

Open the local web URL provided by Streamlit in your browser.

Select the disease to predict, fill in the required fields, and view the results.

Python Libraries and Frameworks Used
Library/Framework	Purpose
streamlit	Web app interface
pandas	Data handling and preprocessing
pickle	Model loading
matplotlib	Data visualization
gtts	Text-to-speech feedback
streamlit-option-menu	Sidebar navigation
os	System operations
Datasets
The models are trained on publicly available datasets from Kaggle for each condition:

Disease	Source	Typical Kaggle Dataset URL Format
Diabetes	Kaggle	kaggle.com/datasets/...
Heart Disease	Kaggle	kaggle.com/datasets/...
Parkinson's Disease	Kaggle	kaggle.com/datasets/...
Cardio Disease	Kaggle	kaggle.com/datasets/...
Hepatitis	Kaggle	kaggle.com/datasets/...
Liver Disease	Kaggle	kaggle.com/datasets/...
Ocular Disease	Kaggle	kaggle.com/datasets/...
Thyroid Disease	Kaggle	kaggle.com/datasets/...
Multiple Sclerosis	Kaggle	kaggle.com/datasets/...
Please ensure you download the datasets individually from Kaggle, as per the README or documentation, and place them in the appropriate folder in your project.

Contact
For questions or contributions, please open an issue or submit a pull request to the repository.
