Multi-Disease Prediction App
📌 Project Summary
The Multi-Disease Prediction App is an interactive web application built using Python and the Streamlit framework. Its main goal is to assist users—patients, healthcare professionals, and enthusiasts—in predicting their risk for multiple diseases with a single interface.

The app leverages machine learning models trained on real medical data to provide risk assessments for:

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

✨ Features
✅ Single Platform: Predicts nine major diseases from one unified app.
✅ Easy-to-Use Interface: Built with Streamlit for responsiveness and clarity.
✅ Machine-Learning Powered: Models trained on reputable Kaggle datasets.
✅ Text-to-Speech Integration: gTTS gives users an audio summary of results.
✅ Beautiful Visuals: Data visualization with Matplotlib for better insights.

🚀 Steps to Run the Application
1️⃣ Clone or Download the Repository
bash
Copy
Edit
git clone https://github.com/MOHAMMEDFAISALSM/multi-disease-prediction-streamlit-app.git
cd multi-disease-prediction-streamlit-app
2️⃣ Install Required Libraries
Make sure you have Python 3.7+ installed.
Install dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install individually:

bash
Copy
Edit
pip install streamlit pandas matplotlib gtts streamlit-option-menu
3️⃣ Download the Datasets
The app uses nine datasets, all of which can be downloaded directly from Kaggle.
Refer to the app’s documentation or Kaggle links for each disease:

Diabetes: Kaggle Dataset

Heart Disease: Kaggle Dataset

Parkinson's Disease: Kaggle Dataset

Cardio Disease: Kaggle Dataset

Hepatitis: Kaggle Dataset

Liver Disease: Kaggle Dataset

Ocular Disease: Kaggle Dataset

Thyroid Disease: Kaggle Dataset

Multiple Sclerosis: Kaggle Dataset

📁 Make sure to place each dataset in the appropriate folder as described in the app.

4️⃣ Run the Streamlit App
From the project root directory, execute:

bash
Copy
Edit
streamlit run multi_disease_pred.py
Example (if your script is named multi_disease_pred.py):

bash
Copy
Edit
streamlit run multi_disease_pred.py
5️⃣ Interact with the Web App
Open the local web URL provided by Streamlit in your browser.

Select the disease to predict.

Fill in the required fields.

Get instant prediction results with an audio summary!

⚙️ Python Libraries and Frameworks Used
Library/Framework	Purpose
streamlit	Web app interface
pandas	Data handling and preprocessing
pickle	Model loading
matplotlib	Data visualization
gtts	Text-to-speech feedback
streamlit-option-menu	Sidebar navigation
os	System operations

📊 Datasets
The models are trained on publicly available datasets from Kaggle for each condition. Please ensure you download them individually and place them correctly.

📬 Contact
For questions or contributions, please open an issue or submit a pull request to the repository.

✅ Enjoy predicting and stay healthy!
