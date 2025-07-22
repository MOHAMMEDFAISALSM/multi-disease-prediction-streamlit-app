# 🧠 Multi-Disease Prediction App

## 📌 Project Summary

The **Multi-Disease Prediction App** is a machine learning-based web application developed using the Python **Streamlit** framework. Its purpose is to help users assess their risk of various diseases by entering commonly available medical data.

This application allows the prediction of the following **nine diseases**:

- Diabetes
- Heart Disease
- Parkinson's Disease
- Cardio Disease
- Hepatitis
- Liver Disease
- Ocular Disease
- Thyroid Disease
- Multiple Sclerosis

Each disease prediction is powered by a dedicated machine learning model trained on respective **Kaggle datasets**. The application provides predictions interactively, with a user-friendly interface and audio feedback.

---

## 🚀 Features

- 🔹 **Multi-disease prediction** in a single app
- 🔹 User-friendly interface built with **Streamlit**
- 🔹 Interactive dashboard with audio support using **gTTS**
- 🔹 Uses real **medical datasets from Kaggle**
- 🔹 Clean visualizations using **Matplotlib**
- 🔹 Sidebar navigation using **streamlit-option-menu**

---

## 🛠️ Technologies Used

| Library / Framework         | Purpose                                   |
|----------------------------|--------------------------------------------|
| `streamlit`                | Building interactive web apps              |
| `pandas`                   | Data manipulation and analysis             |
| `pickle`                   | Loading trained machine learning models    |
| `matplotlib`               | Plotting and visualization                 |
| `gtts`                     | Text-to-speech for result narration        |
| `streamlit-option-menu`    | Customized sidebar menu options            |
| `os`                       | File and system-level operations           |

---

## 📂 Datasets

All datasets used in this project have been sourced from **[Kaggle](https://www.kaggle.com/)**. The models are trained on labeled classification datasets relevant to the following diseases:

| Disease               | Dataset Source |
|----------------------|----------------|
| Diabetes             | Kaggle         |
| Heart Disease        | Kaggle         |
| Parkinson's Disease  | Kaggle         |
| Cardio Disease       | Kaggle         |
| Hepatitis            | Kaggle         |
| Liver Disease        | Kaggle         |
| Ocular Disease       | Kaggle         |
| Thyroid Disease      | Kaggle         |
| Multiple Sclerosis   | Kaggle         |

> ⚠️ **Note**: Please ensure you download the datasets from Kaggle and place them in the appropriate directories as expected by the code.

---

## 📦 Installation & Usage

### 1️⃣ Clone the Repository
git clone https://github.com/MOHAMMEDFAISALSM/multi-disease-prediction-streamlit-app.git

cd multi-disease-prediction-streamlit-app

pip install -r requirements.txt


If `requirements.txt` is not available, install manually:

pip install streamlit pandas matplotlib gtts streamlit-option-menu


### 3️⃣ Run the App

To start the Streamlit application, run the following command in your terminal:

streamlit run multi_disease_pred.py




### 2️⃣ Install the Required Packages

You can install the dependencies via `pip`:


> ✅ **Example**:
> If your file name is `multi_disease_pred.py`, then:
> 
> ```
> streamlit run multi_disease_pred.py
> ```

---

## 📷 Screenshots

*(Optional - You can add screenshots of your UI here to give users a visual tour.)*

---

## 🤝 Contributing

Feel free to submit issues or pull requests to improve this project.

---

## 📫 Contact

Created by **[Mohammed Faisal S M](https://github.com/MOHAMMEDFAISALSM)**  
For feedback or queries, open an issue on the [GitHub repository](https://github.com/MOHAMMEDFAISALSM/multi-disease-prediction-streamlit-app).

---

## ⭐ Show Your Support

If you found this project useful, give it a ⭐ on GitHub for motivation!

## 📚 Research Papers & References

This section highlights leading research that supports the application of machine learning for multi-disease prediction systems. These references strengthen the scientific foundation of the project and reflect advancements in the field.

---

### 🔬 Primary Research Studies

- **Feasible Prediction of Multiple Diseases using Machine Learning**  
  *Ramesh B., Srinivas G., Reddy P.R.P., et al. (2023, E3S Web of Conferences, 430, 01051)*  
  > Developed an automated prediction system for multiple diseases using decision trees, SVM, and random forests, achieving **over 95% mean accuracy** across diseases.

- **Disease Prediction System**  
  *Khiratkar R., Sarpate S., Dhande G., et al. (2024, International Journal of Trend in Scientific Research and Development, 8(5): 1023-1031)*  
  > Integrated genomic data, electronic health records, and environmental factors with machine learning algorithms; reported **92.5% accuracy** in disease susceptibility prediction.

- **Multiple Disease Prediction Using Machine Learning**  
  *Mathew L.S., Fathima S.H.S., Surya T., et al. (2024, IJCRT, 12(5))*  
  > Presented a multi-disease prediction framework using support vector machines for heart disease, diabetes, and Parkinson's; offers detailed model performance analysis.

---

### 🏥 Machine Learning Surveys in Healthcare

- **New horizons in prediction modelling using machine learning in older people's healthcare research**  
  *Stahl D. (2024, PMC, PMID: 39311424)*  
  > A comprehensive overview on supervised and unsupervised machine learning prediction models in healthcare, focusing on model validation and clinical applications.

- **A Comprehensive Survey of Machine Learning in Healthcare**  
  *Banapuram C., Naik A.C., Vanteru M.K., et al. (2024, SSRG IJ Electronics & Communication Engineering, 11(5): 155-169)*  
  > Assessed ML’s role in early disease detection and diagnostics, highlighting improvements in prediction accuracy for heart and liver diseases.

- **The Role Of Machine Learning In Transforming Healthcare: A Systematic Review**  
  *Goswami D. (2024, Non Human Journal, 1(01))*  
  > Reviewed 167 studies showing machine learning’s impact on disease diagnosis, personalized medicine, and healthcare efficiency.

---

### 🧠 Advanced ML & Deep Learning Studies

- **Survey of Deep Learning Techniques for Disease Prediction Based on Omics Data**  
  *Xindi Y., Shusen Z., Hailin Z., et al. (2023, Informatics in Medicine Unlocked)*  
  > Detailed review of deep learning algorithms using genomic, proteomic, and metabolomic data for disease prediction.

- **Time Series Prediction Using Deep Learning Methods in Healthcare**  
  *Morid M.A., Sheng O.R.L., Dunbar J. (2021, arXiv:2108.13461)*  
  > Review of deep learning models for time-series clinical data and their use in healthcare prediction systems.

---

### 📊 Algorithm Performance & Implementation

- **Multi Disease Prediction using Machine Learning Algorithms**  
  *Bharath C., Deekshitha G.P., Deepak M.P., et al. (2024, IJIRSET, 13(6))*  
  > Compared Random Forest, SVM, and Decision Tree algorithms; demonstrates high accuracy and reliability for multi-disease prediction tasks.

- **Multiple Disease Prediction Using Machine Learning**  
  *Parshant & Rathee A. (2023, IRE Journals)*  
  > SVM-based multi-disease prediction system recorded **98.3% accuracy** on heart disease, diabetes, and Parkinson's.

---

### 🌟 Foundational Medical AI Studies

- **Scalable and Accurate Deep Learning with Electronic Health Records**  
  *Rajkomar A., Oren E., Chen K., et al. (2018, NPJ Digital Medicine, 1(1))*  
  > Demonstrated deep learning’s potential on electronic health records for broad clinical predictions across thousands of patients.

- **Dermatologist-level Classification of Skin Cancer with Deep Neural Networks**  
  *Esteva A., Kuprel B., Novoa R.A., et al. (2017, Nature, 542: 115-118)*  
  > AI achieved expert-level accuracy in skin cancer detection using convolutional neural networks.

- **Development and Validation of a Deep Learning Algorithm for Detection of Diabetic Retinopathy**  
  *Gulshan V., Peng L., Coram M., et al. (2016, JAMA, 316(22): 2402-2410)*  
  > Pioneering study in deep learning for automated medical image analysis and diabetic retinopathy detection.

---

### ✍️ Usage of References

- These studies **validate the use of SVM, Random Forest, and Decision Tree algorithms** in multi-disease prediction.  
- **Reported accuracies range from 85% to 98%** for similar multi-disease systems.
- Citations include **peer-reviewed journals and systematic reviews** (2016–2024).
- The selection demonstrates **current relevance, scientific rigor, and diverse algorithmic applications in healthcare**.

---

