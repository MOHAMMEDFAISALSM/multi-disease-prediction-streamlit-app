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

| Library / Framework          | Purpose                                   |
|-----------------------------|-------------------------------------------|
| `streamlit`                 | Building interactive web apps             |
| `pandas`                    | Data manipulation and analysis            |
| `pickle`                    | Loading trained machine learning models   |
| `matplotlib`                | Plotting and visualization                |
| `gtts`                      | Text-to-speech for result narration       |
| `streamlit-option-menu`     | Customized sidebar menu options           |
| `os`                        | File and system-level operations          |

---

## 📂 Datasets

All datasets used in this project have been sourced from **[Kaggle](https://www.kaggle.com/)** and correspond to the listed diseases. Please download and store datasets in the correct folders as expected by the code.

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

> ⚠️ **Note:** Make sure to place the datasets in the proper directories expected by the model loader scripts.

---

## 📦 Installation & Usage

Follow these instructions step-by-step after downloading or cloning the project.

### 1️⃣ Navigate to the Project Folder

Open your terminal and run:
> ```
>cd path/to/Multi_disease_using_streamlit_app
> ```

> Replace `path/to/` with your actual folder path.

### 2️⃣ Install Required Python Packages

If `requirements.txt` is available:
> ```
>pip install -r requirements.txt
> ```

If not available, manually install packages:
> ```
>pip install streamlit pandas matplotlib gtts streamlit-option-menu
> ```

---

### 3️⃣ Run the Application

Make sure you're inside the `Multi_disease_using_streamlit_app` directory and then run:
> ```
>streamlit run multi_disease_pred.py
> ```

> ✅ Example:  
> If your main app file is named differently (e.g., `app.py`), update the filename in the command accordingly:
> 

streamlit run app.py


---

## 📷 Screenshots

*(Optional: You can add screenshots of the interface here)*

---

## 🤝 Contributing

Contributions are welcome!  
You can open issues or submit pull requests to help improve this project.

---

## 📫 Contact

Made by **[Mohammed Faisal S M](https://github.com/MOHAMMEDFAISALSM)**  
For questions or support, visit the [GitHub repository](https://github.com/MOHAMMEDFAISALSM/multi-disease-prediction-streamlit-app/issues).

---

## ⭐ Show Your Support

If you found this helpful, please consider giving it a ⭐ on GitHub.  
Your support is appreciated!

---

## 📚 Research Papers & References

### 🔬 Key Research Studies

- **Feasible Prediction of Multiple Diseases using Machine Learning**  
  *Ramesh B., Srinivas G., Reddy P.R.P., et al. (2023, E3S Web of Conferences, 430, 01051)*  
  > Achieves over **95% mean accuracy** with decision trees, SVM, and random forests.

- **Disease Prediction System**  
  *Khiratkar R., Sarpate S., Dhande G., et al. (2024, IJTSRD, 8(5): 1023-1031)*  
  > Combines electronic health records and genomic data with **92.5% prediction** accuracy.

- **Multiple Disease Prediction Using Machine Learning**  
  *Mathew L.S., Fathima S.H.S., et al. (2024, IJCRT, 12(5))*  
  > Framework for predicting heart disease, diabetes, and Parkinson's using SVM.

### 🧠 Machine Learning in Healthcare

- **Prediction Modelling in Elderly Care**  
  *Stahl D. (2024, PMC, PMID: 39311424)*

- **Comprehensive ML Survey in Healthcare**  
  *Banapuram et al. (2024, SSRG IJECE)*

- **The Role of ML in Transforming Healthcare**  
  *Goswami D. (2024, Non Human Journal)*

### 📊 Deep Learning, Time-Series & Algorithm Evaluation

- **Omics-Based Prediction using Deep Learning**  
  *Xindi Y., et al. (2023, Informatics in Medicine Unlocked)*

- **Time Series Prediction Using Deep Models**  
  *Morid M.A., et al. (2021, arXiv:2108.13461)*

- **Algorithm Comparison for Multi-Disease**  
  *Bharath C., Deekshitha G.P., et al. (2024, IJIRSET, 13(6))*  
  > Compared Random Forest, Decision Tree & SVM.

- **SVM Prediction System**  
  *Parshant, Rathee A. (2023, IRE Journals)*  
  > SVM recorded **98.3% accuracy** for heart disease, diabetes, and Parkinson’s.

### 🏥 Landmark Medical AI Papers

- **Scalable Deep Learning for EHR**  
  *Rajkomar A., et al. (2018, NPJ Digital Med, 1(1))*

- **Skin Cancer Detection with DNN**  
  *Esteva A., et al. (2017, Nature, 542: 115-118)*

- **Diabetic Retinopathy Deep Learning**  
  *Gulshan V., et al. (2016, JAMA, 316(22): 2402–2410)*

---

## 🧾 Reference Usage

- Validates use of SVM, RF, and Decision Trees in multi-disease classification  
- Reported accuracies range: **85%–98.3%**  
- Studies span from **2016 to 2024**, ensuring relevance and credibility  
- Includes **peer-reviewed journals and system reviews**

