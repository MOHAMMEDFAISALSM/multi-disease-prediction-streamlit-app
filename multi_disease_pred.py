import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from gtts import gTTS
import os
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Disease Prediction",
    layout="wide"
)

# --- Load models with error handling ---
try:
    diabetes_model, diabetes_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/diabetes_model_scaler.sav', 'rb'))
    heart_model, heart_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/heart_model_scaler.sav', 'rb'))
    parkinsons_model, parkinsons_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/parkinsons_model_scaler.sav', 'rb'))
    cardio_model, cardio_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/cardio_model_scaler.sav', 'rb'))
    hepatitis_model, hepatitis_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/hepatitis_model_scaler.sav', 'rb'))
    liver_model, liver_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/liver_model_scaler.sav', 'rb'))
    ms_model, ms_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/ms_model_scaler.sav', 'rb'))
    ocular_model, ocular_scaler = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/ocular_model_scaler.sav', 'rb'))
    thyroid_model, thyroid_scaler, thyroid_label_encoders, thyroid_target_le = pickle.load(open(
        'D:/faisal-VS/faisal project/ML project/saved models/thyroid_model_scaler.sav', 'rb'))
except FileNotFoundError as e:
    st.error(f"🚫 Error: Could not find a model file.\nDetails: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    selected = option_menu(
        "Disease Prediction",
        [
            "Home",
            "Diabetes",
            "Heart Disease",
            "Parkinsons",
            "Cardio",
            "Hepatitis",
            "Liver",
            "MS",
            "Ocular",
            "Thyroid"
        ],
        icons=[
            "house",
            "activity",
            "heart",
            "person",
            "heart-pulse",
            "droplet-half",
            "shield-plus",
            "cpu",
            "eye",
            "thermometer"
        ],
        default_index=0
    )

# --- Home Page ---
if selected == 'Home':
    st.title("🩺 Multiple Disease Prediction System")
    st.subheader("Early Detection for a Healthier Tomorrow")

    # Voice intro with gTTS
    welcome_message = (
        "Welcome to the Multiple Disease Prediction System. "
        "Please read the notes carefully and get ready to dive into our models for early health detection."
    )
    tts_file = "welcome_home.mp3"
    if not os.path.exists(tts_file):
        tts = gTTS(welcome_message)
        tts.save(tts_file)
    audio_file = open(tts_file, "rb")
    st.audio(audio_file.read(), format="audio/mp3")

    # Main introduction
    st.write("""
    Welcome to the **Multiple Disease Prediction System**, your AI-powered virtual health assistant.
    This platform helps you assess your risk levels for **nine common diseases** using
    **advanced machine learning models** — all in one place.
    
    ### 🌟 Why Use This Platform?
    - **Instant Results, Personalized Advice, No Waiting:**  
      Get a clinically-inspired assessment and tailored lifestyle recommendations in seconds.
    - **Privacy First:**  
      Your data never leaves your device unless you choose to share it. We prioritize your confidentiality and trust.
    - **Science You Can Trust:**  
      Models are trained on authentic medical datasets for accuracy and reliability.
    """)

    # Platform features
    st.markdown("### ⚡ Unique Features You'll Love")
    st.write("""
    - **Audio Summaries:** Every result comes with a clear, human-like audio explanation—perfect for learning on the go, those with visual difficulties, or anyone preferring to listen.
    - **Visual Health Benchmarks:** See where you stand instantly with graphs and sample reports compared to healthy ranges.
    - **Comprehensive Disease Coverage:** All your major health checks in a single, fast, and secure interface.
    - **Food Guidance From Day One:** Get not only your risk scores, but *practical nutrition tips*—what to eat, what to avoid, and why it matters.
    - **Interactive Learning:** Sample reports, clear “what’s new” highlights, and guided navigation make it easy for any age or background.
    """)

    st.write("""
    ### 🕹️ How To Get the Most From This App
    1. **Be Honest and Accurate:** For best results, input your most recent and reliable test data.
    2. **Use It to Learn:** Discover patterns and indicators that matter for your health. Each prediction explains both the risk and its implications.
    3. **Follow Up Professionally:** Always consult with your doctor—bring your report along for an informed discussion!
    """)

    st.write("""
    ### 💡 Did You Know?
    - AI-powered risk prediction platforms are transforming healthcare globally, enabling earlier intervention and improved patient outcomes.
    - Small dietary changes, guided by personalized recommendations, can reduce the risk of chronic disease progression by up to 30% when followed consistently.
    - Combining audio reports with visual summaries helps users retain health knowledge more effectively than text alone.
    """)

    st.write("""
    ### 👓 Accessibility & Inclusivity
    - Built-in audio ensures support for users with low vision or reading difficulties.
    - Simple language and visuals cater to any age group or educational background.
    - Responsive design means it works on any device: desktop, tablet, or phone.
    """)

    st.write("""
    ### 🏆 Our Commitment
    Your health is our passion. Let technology guide your journey to a happier, healthier, and more informed life—securely, reliably, and with a human touch.

    **Ready to begin?**  
    Select a disease module from the sidebar, and take the next step towards peace of mind and proactive wellness!
    """)

    # Supported diseases
    st.write("""
    **Supported diseases:**
    - Diabetes
    - Heart Disease
    - Parkinson’s Disease
    - Cardiovascular Disease
    - Hepatitis
    - Liver Disease
    - Multiple Sclerosis (MS)
    - Ocular Diseases
    - Thyroid Disorders
    """)

    # Banner Image
    st.image(
        "D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/banner.png",
        use_column_width=True,
        caption="Empowering early detection with Artificial Intelligence"
    )

    # What's New Section
    st.markdown("### ✅ What’s New?")
    st.write("""
    ✨ **Personalized Food Recommendations:**  
    Along with the risk prediction, our system now generates a **sample food prescription** — healthy foods to include and foods to limit — based on your predicted condition.  
    This helps you make small dietary changes today for a healthier tomorrow.
    """)

    # Model accuracy table
    st.markdown("### 📊 Model Accuracy Overview")
    st.write("""
    Our models are tested on real medical datasets. Below are current accuracy scores — new models are being tested and improved every day.

    | Disease            | Train Accuracy | Test Accuracy |
    |--------------------|---------------|--------------|
    | Diabetes           | 78.7%         | 77.3%        |
    | Heart Disease      | 85.85%        | 80.49%       |
    | Parkinson’s        | 89.74%        | 89.74%       |
    | Cardio Disease     | 72.99%        | 72.99%       |
    | Hepatitis          | Coming soon   | Coming soon  |
    | Liver Disease      | Coming soon   | Coming soon  |
    | Multiple Sclerosis | Coming soon   | Coming soon  |
    | Ocular Disease     | Coming soon   | Coming soon  |
    | Thyroid Disorder   | Coming soon   | Coming soon  |
    """)

    # Health marker graph
    st.markdown("### 📈 Example Normal Ranges")
    st.write("""
    Here’s a quick look at healthy normal ranges for common health markers.
    Compare them to your values to better understand your report.
    """)
    st.image(
        "D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/graph_image.png",
        caption="Normal vs Abnormal Health Indicators"
    )

    # Sample report image
    st.markdown("### 📄 Sample Report and Prescription")
    st.write("""
    Your final report includes:
    - **Risk Prediction** (Safe / At Risk)
    - **AI Voice Summary**
    - **Food Prescription** — Example of what to eat more and what to avoid

    Below is an example of how it looks:
    """)
    st.image(
        "D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/report.png",
        caption="Sample Generated Report with Food-wise Tips"
    )

    # How to get started steps
    st.markdown("### ▶️ Get Started in 4 Steps")
    st.write("""
    ✅ **Step 1:** Select a disease module from the sidebar (Diabetes, Heart, Parkinson’s, Cardio, Hepatitis, Liver, MS, Ocular, Thyroid)

    ✅ **Step 2:** Enter your health test values

    ✅ **Step 3:** Click **Generate Result** — our AI will process your data instantly

    ✅ **Step 4:** Listen to the voice summary and check your personalized food prescription for better daily habits
    """)

    # Disclaimer and Developer credit
    st.info("""
    ℹ️ **Disclaimer:** This tool is for **educational and informational use only** and does not replace professional medical advice, diagnosis, or treatment. 
    Always consult your doctor for any health concerns.
    """)

    st.success("""
    👨‍💻 Developed by **Mohammed Faisal**  
    | Powered by Machine Learning & Streamlit | Your Health, Our Priority.
    """)


from gtts import gTTS
import base64

# --- DIABETES PAGE ---
if selected == 'Diabetes':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img1.jpg", use_column_width=True)
    st.title('🩸 Diabetes Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ""

    if st.button('🧪 Generate Diabetes Test Result'):
        try:
            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ])
            input_scaled = diabetes_scaler.transform(input_df)
            diab_prediction = diabetes_model.predict(input_scaled)

            if diab_prediction[0] == 1:
                icon = "⚠️"
                color = "#FF4B4B"
                diab_diagnosis = f"{icon} **{person_name}**, we are sorry to say you are **Diabetic**."
                advice = "⚠️ Please consult a doctor immediately."
                foods = """
                **Recommended Foods:**  
                - Whole grains (oats, brown rice)  
                - Leafy vegetables (spinach, kale)  
                - Nuts and seeds (almonds, flaxseeds)  
                - Lean protein (fish, chicken)

                **Foods to Avoid:**  
                - Sugary drinks & sweets  
                - White bread & refined carbs  
                - Fried foods & processed snacks  
                """
            else:
                icon = "✅"
                color = "#4CAF50"
                diab_diagnosis = f"{icon} **{person_name}**, congratulations! You are **NOT Diabetic**."
                advice = "✅ Stay healthy, eat balanced food & monitor sugar levels regularly."
                foods = """
                **Healthy Eating Tips:**  
                - Keep eating fiber-rich foods  
                - Stay hydrated  
                - Maintain a healthy weight  
                - Exercise regularly

                **Things to Limit:**  
                - Excess processed sugar  
                - Sugary sodas
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("diabetes_result.mp3")
            audio_file = open("diabetes_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(diab_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [2, 110, 70, 20, 80, 25, 0.5, 30]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=30)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food Prescription ---
            st.subheader("🥗 Your Food Prescription")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            It may not be fully accurate — please always consult a qualified doctor for any health concerns.
            """)

            st.markdown("""
            ---
            ## 📌 What is Diabetes?

            Diabetes is a long-term health condition that affects how your body turns food into energy.
            Too much sugar stays in your blood, which can lead to serious health problems.
            
            ✅ **How to Control Diabetes:**
            - Maintain a healthy weight
            - Follow a balanced diet
            - Exercise regularly
            - Monitor your sugar levels
            - Take prescribed medication if needed
            - Visit your doctor regularly
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")

from gtts import gTTS

# --- HEART DISEASE PAGE ---
if selected == 'Heart Disease':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img2.jpg", use_column_width=True)
    st.title('❤️ Heart Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_heart")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)')
    with col3:
        cp = st.text_input('Chest Pain Type (0-3)')
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    with col1:
        restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    with col1:
        oldpeak = st.text_input('Oldpeak (ST depression induced by exercise)')
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
    with col3:
        ca = st.text_input('Number of Major Vessels (0-3) colored by fluoroscopy')
    with col1:
        thal = st.text_input('Thal (1 = normal; 2 = fixed defect; 3 = reversable defect)')

    heart_diagnosis = ""

    if st.button('🧪 Generate Heart Disease Test Result'):
        try:
            input_data = [
                float(age),
                float(sex),
                float(cp),
                float(trestbps),
                float(chol),
                float(fbs),
                float(restecg),
                float(thalach),
                float(exang),
                float(oldpeak),
                float(slope),
                float(ca),
                float(thal)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'age', 'sex', 'cp', 'trestbps', 'chol',
                'fbs', 'restecg', 'thalach', 'exang',
                'oldpeak', 'slope', 'ca', 'thal'
            ])

            input_scaled = heart_scaler.transform(input_df)
            heart_prediction = heart_model.predict(input_scaled)

            if heart_prediction[0] == 1:
                icon = "⚠️"
                color = "#FF4B4B"
                heart_diagnosis = f"{icon} **{person_name}**, we are sorry to say you are **at risk of Heart Disease**."
                advice = "⚠️ Please consult a cardiologist as soon as possible."
                foods = """
                **Recommended Foods:**  
                - Fresh fruits & vegetables  
                - Whole grains  
                - Lean protein (fish, poultry)  
                - Nuts & seeds  
                - Olive oil

                **Foods to Avoid:**  
                - Processed meats  
                - Excess salt & fried food  
                - Sugary drinks  
                - Trans fats & excess saturated fats
                """
            else:
                icon = "✅"
                color = "#4CAF50"
                heart_diagnosis = f"{icon} **{person_name}**, congratulations! You are **NOT at risk of Heart Disease**."
                advice = "✅ Keep your heart healthy with good food, exercise, and routine checkups."
                foods = """
                **Healthy Heart Tips:**  
                - Stay active  
                - Eat fiber-rich foods  
                - Keep cholesterol under control  
                - Don’t smoke  
                - Get regular check-ups

                **Things to Limit:**  
                - Processed junk food  
                - Smoking & excess alcohol  
                - High sodium foods
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("heart_result.mp3")
            audio_file = open("heart_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(heart_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [55, 1, 0, 120, 200, 0, 1, 150, 0, 1, 1, 0, 2]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Heart Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=30)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food Prescription ---
            st.subheader("🥗 Your Heart-Healthy Food Prescription")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            It may not be fully accurate — please always consult a qualified doctor for any health concerns.
            """)

            st.markdown("""
            ---
            ## ❤️ What is Heart Disease?

            Heart disease refers to various conditions that affect the heart's structure and function.
            The most common type is coronary artery disease, which can lead to heart attacks.

            ✅ **How to Keep Your Heart Healthy:**
            - Eat a balanced diet rich in fiber & healthy fats  
            - Stay physically active  
            - Control blood pressure & cholesterol  
            - Quit smoking  
            - Manage stress  
            - Get regular screenings
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")




# --- PARKINSON'S DISEASE PAGE ---
if selected == 'Parkinsons':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img3.jpg", use_column_width=True)
    st.title('🧠 Parkinson’s Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_parkinsons")

    # 5 columns layout
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')
    with col4:
        jitter_percent = st.text_input('MDVP:Jitter(%)')
    with col5:
        jitter_abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        rap = st.text_input('MDVP:RAP')
    with col2:
        ppq = st.text_input('MDVP:PPQ')
    with col3:
        ddp = st.text_input('Jitter:DDP')
    with col4:
        shimmer = st.text_input('MDVP:Shimmer')
    with col5:
        shimmer_db = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        apq3 = st.text_input('Shimmer:APQ3')
    with col2:
        apq5 = st.text_input('Shimmer:APQ5')
    with col3:
        apq = st.text_input('MDVP:APQ')
    with col4:
        dda = st.text_input('Shimmer:DDA')
    with col5:
        nhr = st.text_input('NHR')

    with col1:
        hnr = st.text_input('HNR')
    with col2:
        rpde = st.text_input('RPDE')
    with col3:
        dfa = st.text_input('DFA')
    with col4:
        spread1 = st.number_input('spread1', min_value=-10.0, max_value=0.0, value=-5.0)
    with col5:
        spread2 = st.number_input('spread2', min_value=-10.0, max_value=10.0, value=0.0)

    with col1:
        d2 = st.text_input('D2')
    with col2:
        ppe = st.text_input('PPE')

    parkinsons_diagnosis = ""

    if st.button('🧪 Generate Parkinson’s Test Result'):
        try:
            input_data = [
                float(fo), float(fhi), float(flo),
                float(jitter_percent), float(jitter_abs), float(rap),
                float(ppq), float(ddp), float(shimmer), float(shimmer_db),
                float(apq3), float(apq5), float(apq), float(dda),
                float(nhr), float(hnr), float(rpde), float(dfa),
                float(spread1), float(spread2), float(d2), float(ppe)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
                'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
                'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
                'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA',
                'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2',
                'D2', 'PPE'
            ])

            input_scaled = parkinsons_scaler.transform(input_df)
            parkinsons_prediction = parkinsons_model.predict(input_scaled)

            if parkinsons_prediction[0] == 1:
                icon = "⚠️"
                color = "#FF4B4B"
                parkinsons_diagnosis = f"{icon} **{person_name}**, it seems you **may have Parkinson’s Disease**."
                advice = "⚠️ Please consult a neurologist for further tests and medical advice."
                foods = """
                **Recommended Tips:**  
                - Eat antioxidant-rich foods (berries, broccoli)  
                - Include omega-3 fatty acids (salmon, flaxseeds)  
                - Stay hydrated  
                - Maintain a balanced diet with fiber

                **Things to Avoid:**  
                - Excess saturated fats  
                - Processed sugary snacks  
                - Excess alcohol
                """
            else:
                icon = "✅"
                color = "#4CAF50"
                parkinsons_diagnosis = f"{icon} **{person_name}**, good news! You are **NOT at risk of Parkinson’s Disease**."
                advice = "✅ Stay healthy, keep a balanced diet and do regular check-ups."
                foods = """
                **Healthy Lifestyle Tips:**  
                - Eat a Mediterranean-style diet  
                - Include plenty of fruits & vegetables  
                - Stay active with light exercises  
                - Get enough sleep

                **Things to Limit:**  
                - High-fat processed foods  
                - Smoking
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("parkinsons_result.mp3")
            audio_file = open("parkinsons_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(parkinsons_diagnosis)
            st.info(advice)

            # --- Summary Table (Example Normal) ---
            normal_values = [
                150, 200, 100, 0.005, 0.00005, 0.002,
                0.003, 0.01, 0.02, 0.2, 0.01, 0.02,
                0.03, 0.01, 0.02, 20, 0.4, 0.75, -6, 0.1,
                2.5, 0.3
            ]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Parkinson’s Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food/Wellness Tips ---
            st.subheader("🥗 Your Health & Diet Tips")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            Please consult a medical professional for any real health concerns.
            """)

            st.markdown("""
            ---
            ## 🧠 What is Parkinson’s Disease?

            Parkinson’s Disease is a brain disorder that leads to shaking, stiffness, and difficulty with walking, balance, and coordination.

            ✅ **How to Manage Parkinson’s:**
            - Follow doctor-prescribed treatment
            - Eat a healthy, balanced diet
            - Stay active with exercises that improve mobility
            - Join support groups for emotional health
            - Get regular check-ups
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")



# --- CARDIOVASCULAR DISEASE PAGE ---
if selected == 'Cardio':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img4.jpg", use_column_width=True)
    st.title('❤️ Cardiovascular Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_cardio")

    col1, col2, col3 = st.columns(3)

    with col1:
        age_years = st.text_input('Age (years)')
    with col2:
        gender = st.selectbox('Gender', ['1 - Female', '2 - Male'])
    with col3:
        ap_hi = st.text_input('Systolic Blood Pressure (ap_hi)')

    with col1:
        ap_lo = st.text_input('Diastolic Blood Pressure (ap_lo)')
    with col2:
        cholesterol = st.selectbox('Cholesterol Level', ['1 - Normal', '2 - Above Normal', '3 - Well Above Normal'])
    with col3:
        gluc = st.selectbox('Glucose Level', ['1 - Normal', '2 - Above Normal', '3 - Well Above Normal'])

    with col1:
        bmi = st.text_input('Body Mass Index (BMI)')

    cardio_diagnosis = ""

    if st.button('🧪 Generate Cardio Test Result'):
        try:
            input_data = [
                float(age_years),
                int(gender.split(' - ')[0]),
                float(ap_hi),
                float(ap_lo),
                int(cholesterol.split(' - ')[0]),
                int(gluc.split(' - ')[0]),
                float(bmi)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'age', 'gender', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'bmi'
            ])

            input_scaled = cardio_scaler.transform(input_df)
            cardio_prediction = cardio_model.predict(input_scaled)

            if cardio_prediction[0] == 1:
                icon = "⚠️"
                cardio_diagnosis = f"{icon} **{person_name}**, you may be at risk of **Cardiovascular Disease**."
                advice = "⚠️ Please consult a cardiologist for further examination and professional advice."
                foods = """
                **Heart-Healthy Tips:**  
                - Eat fruits, vegetables & whole grains  
                - Choose lean proteins (fish, poultry)  
                - Reduce salt and saturated fats  
                - Exercise regularly

                **Foods to Avoid:**  
                - Deep-fried foods  
                - Excess red meat  
                - Sugary drinks  
                - Processed snacks
                """
            else:
                icon = "✅"
                cardio_diagnosis = f"{icon} **{person_name}**, congratulations! You are **NOT at risk** of Cardiovascular Disease."
                advice = "✅ Keep your heart healthy with balanced meals and active lifestyle."
                foods = """
                **Healthy Heart Tips:**  
                - Stay active with daily walks  
                - Eat fiber-rich foods  
                - Drink enough water  
                - Avoid smoking

                **Things to Limit:**  
                - High-sodium foods  
                - Sugary processed foods
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("cardio_result.mp3")
            audio_file = open("cardio_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(cardio_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [30, 1, 120, 80, 1, 1, 23]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Cardio Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=30)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Heart Tips ---
            st.subheader("🥗 Your Heart Health Tips")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            It may not be fully accurate — please always consult a qualified doctor for any health concerns.
            """)

            st.markdown("""
            ---
            ## ❤️ What is Cardiovascular Disease?

            Cardiovascular Disease affects the heart and blood vessels.  
            High blood pressure, cholesterol, and unhealthy lifestyle can increase risk.

            ✅ **How to Prevent Cardio Disease:**
            - Eat balanced meals low in salt and fat
            - Exercise at least 30 mins daily
            - Maintain healthy weight
            - Quit smoking
            - Visit your doctor for regular check-ups
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")


# --- HEPATITIS PREDICTION PAGE ---
if selected == 'Hepatitis':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img5.jpg", use_column_width=True)
    st.title('🧬 Hepatitis Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_hepatitis")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')
    with col2:
        sex = st.selectbox('Sex', ['1 - Male', '2 - Female'])
    with col3:
        steroid = st.selectbox('Steroid', ['1 - Yes', '2 - No'])

    with col1:
        antivirals = st.selectbox('Antivirals', ['1 - Yes', '2 - No'])
    with col2:
        fatigue = st.selectbox('Fatigue', ['1 - Yes', '2 - No'])
    with col3:
        malaise = st.selectbox('Malaise', ['1 - Yes', '2 - No'])

    with col1:
        anorexia = st.selectbox('Anorexia', ['1 - Yes', '2 - No'])
    with col2:
        liver_big = st.selectbox('Liver Big', ['1 - Yes', '2 - No'])
    with col3:
        liver_firm = st.selectbox('Liver Firm', ['1 - Yes', '2 - No'])

    with col1:
        spleen_palpable = st.selectbox('Spleen Palpable', ['1 - Yes', '2 - No'])
    with col2:
        spiders = st.selectbox('Spiders', ['1 - Yes', '2 - No'])
    with col3:
        ascites = st.selectbox('Ascites', ['1 - Yes', '2 - No'])

    with col1:
        varices = st.selectbox('Varices', ['1 - Yes', '2 - No'])
    with col2:
        bilirubin = st.text_input('Bilirubin')
    with col3:
        alk_phosphate = st.text_input('Alkaline Phosphate')

    with col1:
        sgot = st.text_input('SGOT')
    with col2:
        albumin = st.text_input('Albumin')
    with col3:
        protime = st.text_input('Prothrombin Time')

    with col1:
        histology = st.selectbox('Histology', ['1 - Yes', '2 - No'])

    hep_diagnosis = ""

    if st.button('🧪 Generate Hepatitis Test Result'):
        try:
            input_data = [
                float(age),
                int(sex.split(' - ')[0]),
                int(steroid.split(' - ')[0]),
                int(antivirals.split(' - ')[0]),
                int(fatigue.split(' - ')[0]),
                int(malaise.split(' - ')[0]),
                int(anorexia.split(' - ')[0]),
                int(liver_big.split(' - ')[0]),
                int(liver_firm.split(' - ')[0]),
                int(spleen_palpable.split(' - ')[0]),
                int(spiders.split(' - ')[0]),
                int(ascites.split(' - ')[0]),
                int(varices.split(' - ')[0]),
                float(bilirubin),
                float(alk_phosphate),
                float(sgot),
                float(albumin),
                float(protime),
                int(histology.split(' - ')[0])
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'age', 'sex', 'steroid', 'antivirals', 'fatigue', 'malaise',
                'anorexia', 'liver_big', 'liver_firm', 'spleen_palpable',
                'spiders', 'ascites', 'varices', 'bilirubin',
                'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology'
            ])

            input_scaled = hepatitis_scaler.transform(input_df)
            hep_prediction = hepatitis_model.predict(input_scaled)

            if hep_prediction[0] == 1:
                icon = "⚠️"
                hep_diagnosis = f"{icon} **{person_name}**, you may have **Hepatitis Disease**."
                advice = "⚠️ Please consult a liver specialist for immediate treatment advice."
                foods = """
                **Liver-Friendly Tips:**  
                - Eat fresh fruits & vegetables  
                - Drink plenty of water  
                - Include lean proteins & whole grains  
                - Avoid alcohol completely

                **Foods to Avoid:**  
                - Fried and greasy foods  
                - Excess salt  
                - Processed meats  
                - Sugary desserts & soda
                """
            else:
                icon = "✅"
                hep_diagnosis = f"{icon} **{person_name}**, you are **NOT at risk** of Hepatitis Disease."
                advice = "✅ Maintain good liver health by following a healthy lifestyle."
                foods = """
                **Liver Health Tips:**  
                - Stay hydrated  
                - Eat balanced meals  
                - Avoid excessive medication use  
                - Get vaccinated for hepatitis if needed

                **Things to Limit:**  
                - Alcohol  
                - High-fat processed foods
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("hepatitis_result.mp3")
            audio_file = open("hepatitis_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(hep_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [35, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1.0, 85.0, 45.0, 4.0, 60.0, 1]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Hepatitis Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food Tips ---
            st.subheader("🥗 Your Liver Health Tips")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            Always consult a qualified doctor for any real health concerns.
            """)

            st.markdown("""
            ---
            ## 🧬 What is Hepatitis?

            Hepatitis is inflammation of the liver usually caused by a viral infection.  
            If untreated, it can cause serious liver damage.

            ✅ **How to Manage Hepatitis:**
            - Get timely medical treatment  
            - Eat a liver-friendly diet  
            - Avoid alcohol  
            - Take prescribed medication  
            - Regular check-ups with your doctor
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")


# --- LIVER DISEASE PREDICTION PAGE ---
if selected == 'Liver':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img8.jpg", use_column_width=True)
    st.title('🧪 Liver Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_liver")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age of the Patient')
    with col2:
        Gender = st.selectbox('Gender of the Patient', ['1 - Male', '2 - Female'])
    with col3:
        Total_Bilirubin = st.text_input('Total Bilirubin')

    with col1:
        Direct_Bilirubin = st.text_input('Direct Bilirubin')
    with col2:
        Alkphos = st.text_input('Alkaline Phosphotase')
    with col3:
        Sgpt = st.text_input('Alamine Aminotransferase (Sgpt)')

    with col1:
        Sgot = st.text_input('Aspartate Aminotransferase (Sgot)')
    with col2:
        Total_Proteins = st.text_input('Total Proteins')
    with col3:
        ALB = st.text_input('Albumin')

    with col1:
        AG_Ratio = st.text_input('A/G Ratio (Albumin and Globulin Ratio)')

    liver_diagnosis = ""

    if st.button('🧪 Generate Liver Disease Test Result'):
        try:
            input_data = [
                float(Age),
                int(Gender.split(' - ')[0]),
                float(Total_Bilirubin),
                float(Direct_Bilirubin),
                float(Alkphos),
                float(Sgpt),
                float(Sgot),
                float(Total_Proteins),
                float(ALB),
                float(AG_Ratio)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
                'Alkphos', 'Sgpt', 'Sgot', 'Total_Proteins',
                'ALB', 'AG_Ratio'
            ])

            input_scaled = liver_scaler.transform(input_df)
            liver_prediction = liver_model.predict(input_scaled)

            if liver_prediction[0] == 1:
                icon = "⚠️"
                liver_diagnosis = f"{icon} **{person_name}**, you may have **Liver Disease**."
                advice = "⚠️ Please consult a liver specialist for further diagnosis and treatment."
                foods = """
                **Liver-Friendly Diet Tips:**  
                - Eat fresh fruits & vegetables  
                - Include lean protein (fish, chicken, legumes)  
                - Drink plenty of water  
                - Use healthy fats (olive oil, nuts)

                **Foods to Avoid:**  
                - Alcohol  
                - Fried & fatty foods  
                - Excess salt  
                - Sugary desserts & sodas
                """
            else:
                icon = "✅"
                liver_diagnosis = f"{icon} **{person_name}**, you are **NOT at risk** of Liver Disease."
                advice = "✅ Keep your liver healthy by eating well and avoiding harmful substances."
                foods = """
                **Healthy Liver Tips:**  
                - Stay hydrated  
                - Eat balanced, nutrient-rich meals  
                - Avoid alcohol or drink in moderation  
                - Exercise regularly

                **Things to Limit:**  
                - Highly processed foods  
                - Deep-fried foods  
                - Excess sugar
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("liver_result.mp3")
            audio_file = open("liver_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(liver_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [45, 1, 1.0, 0.3, 90, 35, 35, 7.0, 4.0, 1.2]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Liver Disease Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food Tips ---
            st.subheader("🥗 Your Food & Lifestyle Tips")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            Please consult a qualified doctor for any real health concerns.
            """)

            st.markdown("""
            ---
            ## 🧬 What is Liver Disease?

            Liver disease refers to any condition that damages or affects the liver.  
            It can be caused by infections, alcohol use, or genetic factors.

            ✅ **How to Protect Your Liver:**
            - Avoid alcohol or drink responsibly
            - Eat a balanced, healthy diet
            - Exercise regularly
            - Get regular medical check-ups
            - Take medications only as prescribed
            """)

        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")


# --- MULTIPLE SCLEROSIS PREDICTION PAGE ---
if selected == 'MS':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img9.jpg", use_column_width=True)
    st.title('🧠 Multiple Sclerosis Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_ms")

    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.selectbox('Gender', ['0 - Female', '1 - Male'])
    with col2:
        Age = st.text_input('Age')
    with col3:
        Schooling = st.text_input('Years of Schooling')

    with col1:
        Breastfeeding = st.selectbox('Breastfeeding', ['0 - No', '1 - Yes'])
    with col2:
        Varicella = st.selectbox('Varicella (Chickenpox)', ['0 - No', '1 - Yes'])
    with col3:
        Initial_Symptom = st.selectbox('Initial Symptom', ['1 - Monosymptomatic', '2 - Polysymptomatic'])

    with col1:
        Mono_or_Polysymptomatic = st.selectbox('Mono or Polysymptomatic', ['0 - Mono', '1 - Poly'])
    with col2:
        Oligoclonal_Bands = st.selectbox('Oligoclonal Bands', ['0 - Negative', '1 - Positive'])
    with col3:
        LLSSEP = st.selectbox('LLSSEP', ['0 - Abnormal', '1 - Normal'])

    with col1:
        ULSSEP = st.selectbox('ULSSEP', ['0 - Abnormal', '1 - Normal'])
    with col2:
        VEP = st.selectbox('VEP', ['0 - Abnormal', '1 - Normal'])
    with col3:
        BAEP = st.selectbox('BAEP', ['0 - Abnormal', '1 - Normal'])

    with col1:
        Periventricular_MRI = st.selectbox('Periventricular MRI', ['0 - Abnormal', '1 - Normal'])
    with col2:
        Cortical_MRI = st.selectbox('Cortical MRI', ['0 - Abnormal', '1 - Normal'])
    with col3:
        Infratentorial_MRI = st.selectbox('Infratentorial MRI', ['0 - Abnormal', '1 - Normal'])

    with col1:
        Spinal_Cord_MRI = st.selectbox('Spinal Cord MRI', ['0 - Abnormal', '1 - Normal'])
    with col2:
        Initial_EDSS = st.text_input('Initial EDSS')
    with col3:
        Final_EDSS = st.text_input('Final EDSS')

    ms_diagnosis = ""

    if st.button('🧪 Generate MS Test Result'):
        try:
            input_data = [
                int(Gender.split(' - ')[0]),
                float(Age),
                float(Schooling),
                int(Breastfeeding.split(' - ')[0]),
                int(Varicella.split(' - ')[0]),
                int(Initial_Symptom.split(' - ')[0]),
                int(Mono_or_Polysymptomatic.split(' - ')[0]),
                int(Oligoclonal_Bands.split(' - ')[0]),
                int(LLSSEP.split(' - ')[0]),
                int(ULSSEP.split(' - ')[0]),
                int(VEP.split(' - ')[0]),
                int(BAEP.split(' - ')[0]),
                int(Periventricular_MRI.split(' - ')[0]),
                int(Cortical_MRI.split(' - ')[0]),
                int(Infratentorial_MRI.split(' - ')[0]),
                int(Spinal_Cord_MRI.split(' - ')[0]),
                float(Initial_EDSS),
                float(Final_EDSS)
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'Gender', 'Age', 'Schooling', 'Breastfeeding', 'Varicella',
                'Initial_Symptom', 'Mono_or_Polysymptomatic', 'Oligoclonal_Bands',
                'LLSSEP', 'ULSSEP', 'VEP', 'BAEP',
                'Periventricular_MRI', 'Cortical_MRI', 'Infratentorial_MRI',
                'Spinal_Cord_MRI', 'Initial_EDSS', 'Final_EDSS'
            ])

            input_scaled = ms_scaler.transform(input_df)
            ms_prediction = ms_model.predict(input_scaled)

            if ms_prediction[0] == 1:
                icon = "⚠️"
                ms_diagnosis = f"{icon} **{person_name}**, you may have **Multiple Sclerosis (MS)**."
                advice = "⚠️ Please consult a neurologist for detailed testing and personalized treatment."
                foods = """
                **MS Management Tips:**  
                - Eat omega-3 rich foods (salmon, walnuts)  
                - Include plenty of fruits & vegetables  
                - Stay active with low-impact exercise (yoga, swimming)  
                - Get enough vitamin D (safe sun exposure, supplements)

                **Things to Avoid:**  
                - Smoking  
                - Processed foods high in saturated fat  
                - Excess alcohol
                """
            else:
                icon = "✅"
                ms_diagnosis = f"{icon} **{person_name}**, you are **NOT at risk** for Multiple Sclerosis."
                advice = "✅ Stay healthy with a balanced diet and routine health checks."
                foods = """
                **Healthy Lifestyle Tips:**  
                - Eat a Mediterranean diet  
                - Stay physically active  
                - Maintain a healthy sleep cycle  
                - Manage stress effectively

                **Things to Limit:**  
                - Smoking  
                - Highly processed foods
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("ms_result.mp3")
            audio_file = open("ms_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(ms_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [1, 35, 16, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1.0, 1.5]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your MS Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal Levels', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Ranges')
            ax.legend()
            st.pyplot(fig)

            # --- Food & Lifestyle Tips ---
            st.subheader("🥗 Your Lifestyle & Diet Tips")
            st.markdown(foods)

            st.info("""
            ℹ️ **Note:** This result is for **educational/testing purposes only**.  
            Please consult a qualified neurologist for medical confirmation.
            """)

            st.markdown("""
            ---
            ## 🧩 What is Multiple Sclerosis (MS)?

            MS is an autoimmune disease that affects the brain and spinal cord.  
            It can cause vision, balance, and muscle control problems.

            ✅ **How to Manage MS:**
            - Follow your doctor’s treatment plan
            - Eat nutritious, anti-inflammatory foods
            - Stay active with gentle exercises
            - Manage stress and mental well-being
            """)
        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")

# --- OCULAR DISEASE PREDICTION PAGE ---
if selected == 'Ocular':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img7.jpg", use_column_width=True)
    st.title('👁️ Ocular Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_ocular")

    col1, col2, col3 = st.columns(3)

    with col1:
        Patient_Age = st.text_input('Patient Age')
    with col2:
        Patient_Sex = st.selectbox('Patient Sex', ['0 - Female', '1 - Male'])
    with col3:
        Left_Fundus = st.selectbox('Left Fundus Present?', ['0 - No', '1 - Yes'])

    with col1:
        Right_Fundus = st.selectbox('Right Fundus Present?', ['0 - No', '1 - Yes'])
    with col2:
        Left_Diagnostic_Keywords = st.selectbox('Left Diagnostic Keywords Present?', ['0 - No', '1 - Yes'])
    with col3:
        Right_Diagnostic_Keywords = st.selectbox('Right Diagnostic Keywords Present?', ['0 - No', '1 - Yes'])

    with col1:
        N = st.selectbox('N (Nerve issues)', ['0 - No', '1 - Yes'])
    with col2:
        D = st.selectbox('D (Diabetes related)', ['0 - No', '1 - Yes'])
    with col3:
        G = st.selectbox('G (Glaucoma)', ['0 - No', '1 - Yes'])

    with col1:
        C = st.selectbox('C (Cataract)', ['0 - No', '1 - Yes'])
    with col2:
        A = st.selectbox('A (Age-related)', ['0 - No', '1 - Yes'])
    with col3:
        H = st.selectbox('H (Hypertension related)', ['0 - No', '1 - Yes'])

    with col1:
        M = st.selectbox('M (Macular issues)', ['0 - No', '1 - Yes'])
    with col2:
        O = st.selectbox('O (Other conditions)', ['0 - No', '1 - Yes'])

    ocular_diagnosis = ""

    if st.button('🧪 Generate Ocular Disease Result'):
        try:
            input_data = [
                float(Patient_Age),
                int(Patient_Sex.split(' - ')[0]),
                int(Left_Fundus.split(' - ')[0]),
                int(Right_Fundus.split(' - ')[0]),
                int(Left_Diagnostic_Keywords.split(' - ')[0]),
                int(Right_Diagnostic_Keywords.split(' - ')[0]),
                int(N.split(' - ')[0]),
                int(D.split(' - ')[0]),
                int(G.split(' - ')[0]),
                int(C.split(' - ')[0]),
                int(A.split(' - ')[0]),
                int(H.split(' - ')[0]),
                int(M.split(' - ')[0]),
                int(O.split(' - ')[0])
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'Patient Age', 'Patient Sex', 'Left-Fundus', 'Right-Fundus',
                'Left-Diagnostic Keywords', 'Right-Diagnostic Keywords',
                'N', 'D', 'G', 'C', 'A', 'H', 'M', 'O'
            ])

            input_scaled = ocular_scaler.transform(input_df)
            ocular_prediction = ocular_model.predict(input_scaled)

            if ocular_prediction[0] == 1:
                icon = "⚠️"
                ocular_diagnosis = f"{icon} **{person_name}**, signs of an **Ocular Disease** detected."
                advice = "⚠️ Please consult an ophthalmologist immediately for detailed eye tests."
                care = """
                **Recommended Care:**  
                - Schedule a full eye exam  
                - Eat leafy greens & fish rich in omega-3  
                - Wear UV-protective sunglasses  
                - Control blood sugar & blood pressure

                **Avoid:**  
                - Smoking  
                - Excess screen time without breaks  
                - Skipping routine eye checkups
                """
            else:
                icon = "✅"
                ocular_diagnosis = f"{icon} **{person_name}**, no signs of an **Ocular Disease** detected."
                advice = "✅ Keep caring for your eyes with healthy habits & routine eye exams."
                care = """
                **Good Eye Health Tips:**  
                - Eat a balanced diet rich in antioxidants  
                - Take breaks when using screens (20-20-20 rule)  
                - Wear protective eyewear outdoors  
                - Stay hydrated & manage chronic conditions

                **Things to Avoid:**  
                - Smoking  
                - Excessive rubbing of eyes
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("ocular_result.mp3")
            audio_file = open("ocular_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(ocular_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [35, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Ocular Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Reference')
            ax.legend()
            st.pyplot(fig)

            # --- Care Tips ---
            st.subheader("👓 Your Eye Care Tips")
            st.markdown(care)

            st.info("""
            ℹ️ **Note:** This prediction is for **educational/testing purposes only**.  
            Always consult a qualified eye specialist for diagnosis and treatment.
            """)

            st.markdown("""
            ---
            ## 👁️ What are Ocular Diseases?

            Ocular diseases include a range of eye conditions that may affect vision.  
            Early detection and proper care help prevent vision loss.

            ✅ **How to Protect Your Eyes:**
            - Have regular eye exams
            - Eat a balanced, eye-friendly diet
            - Wear sunglasses with UV protection
            - Limit screen time & take breaks
            """)
        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")

# --- THYROID DISEASE PREDICTION PAGE ---
if selected == 'Thyroid':
    st.image("D:/faisal-VS/faisal project/ML project/Multi_disease_using_streamlit_app/img6.jpg", use_column_width=True)
    st.title('🦋 Thyroid Disease Prediction using Machine Learning')

    person_name = st.text_input('🧑 Your Name (Full)', key="name_thyroid")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.text_input('Age')
    with col2:
        Gender = st.selectbox('Gender', ['Male', 'Female'])
    with col3:
        Smoking = st.selectbox('Currently Smoking?', ['No', 'Yes'])

    with col1:
        Hx_Smoking = st.selectbox('History of Smoking?', ['No', 'Yes'])
    with col2:
        Hx_Radiotherapy = st.selectbox('History of Radiotherapy?', ['No', 'Yes'])
    with col3:
        Thyroid_Function = st.selectbox('Thyroid Function', ['Normal', 'Hypothyroidism', 'Hyperthyroidism'])

    with col1:
        Physical_Examination = st.selectbox('Physical Examination Result', ['Normal', 'Abnormal'])
    with col2:
        Adenopathy = st.selectbox('Adenopathy Present?', ['No', 'Yes'])
    with col3:
        Pathology = st.selectbox('Pathology Type', ['Benign', 'Malignant'])

    with col1:
        Focality = st.selectbox('Focality', ['Unifocal', 'Multifocal'])
    with col2:
        Risk = st.selectbox('Risk', ['Low', 'Intermediate', 'High'])
    with col3:
        T = st.selectbox('T Stage', ['T1', 'T2', 'T3', 'T4'])

    with col1:
        N = st.selectbox('N Stage', ['N0', 'N1'])
    with col2:
        M = st.selectbox('M Stage', ['M0', 'M1'])
    with col3:
        Stage = st.selectbox('Overall Stage', ['Stage I', 'Stage II', 'Stage III', 'Stage IV'])

    with col1:
        Response = st.selectbox('Treatment Response', ['Excellent', 'Indeterminate', 'Biochemical Incomplete', 'Structural Incomplete'])
    with col2:
        Recurred = st.selectbox('Has it Recurred?', ['No', 'Yes'])

    thyroid_diagnosis = ""

    if st.button('🧪 Generate Thyroid Test Result'):
        try:
            # Convert all categorical using your label encoders
            input_data = [
                float(Age),
                thyroid_label_encoders['Gender'].transform([Gender])[0],
                thyroid_label_encoders['Smoking'].transform([Smoking])[0],
                thyroid_label_encoders['Hx Smoking'].transform([Hx_Smoking])[0],
                thyroid_label_encoders['Hx Radiothreapy'].transform([Hx_Radiotherapy])[0],
                thyroid_label_encoders['Thyroid Function'].transform([Thyroid_Function])[0],
                thyroid_label_encoders['Physical Examination'].transform([Physical_Examination])[0],
                thyroid_label_encoders['Adenopathy'].transform([Adenopathy])[0],
                thyroid_label_encoders['Pathology'].transform([Pathology])[0],
                thyroid_label_encoders['Focality'].transform([Focality])[0],
                thyroid_label_encoders['Risk'].transform([Risk])[0],
                thyroid_label_encoders['T'].transform([T])[0],
                thyroid_label_encoders['N'].transform([N])[0],
                thyroid_label_encoders['M'].transform([M])[0],
                thyroid_label_encoders['Stage'].transform([Stage])[0],
                thyroid_label_encoders['Response'].transform([Response])[0],
                thyroid_label_encoders['Recurred'].transform([Recurred])[0]
            ]

            input_df = pd.DataFrame([input_data], columns=[
                'Age', 'Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy',
                'Thyroid Function', 'Physical Examination', 'Adenopathy',
                'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response', 'Recurred'
            ])

            input_scaled = thyroid_scaler.transform(input_df)
            thyroid_prediction = thyroid_model.predict(input_scaled)
            thyroid_label = thyroid_target_le.inverse_transform(thyroid_prediction)[0]

            if thyroid_label == 'Disease':
                icon = "⚠️"
                thyroid_diagnosis = f"{icon} **{person_name}**, signs of **Thyroid Disease** detected."
                advice = "⚠️ Please consult an endocrinologist for confirmation and treatment."
                care = """
                **Recommended Care:**  
                - Get TSH, T3, T4 levels tested regularly  
                - Maintain a balanced iodine intake  
                - Follow your doctor's medication plan  
                - Eat a healthy diet with enough protein

                **Avoid:**  
                - Self-medicating with supplements  
                - Ignoring symptoms like swelling, fatigue, sudden weight changes
                """
            else:
                icon = "✅"
                thyroid_diagnosis = f"{icon} **{person_name}**, no signs of **Thyroid Disease** detected."
                advice = "✅ Keep up regular check-ups and a healthy lifestyle."
                care = """
                **Healthy Thyroid Tips:**  
                - Eat iodine-rich foods like fish, eggs, and dairy  
                - Avoid excessive soy if you have thyroid issues  
                - Get regular check-ups if you have a family history  
                - Manage stress and sleep well
                """

            # --- Speak using gTTS ---
            tts = gTTS(advice)
            tts.save("thyroid_result.mp3")
            audio_file = open("thyroid_result.mp3", "rb")
            st.audio(audio_file.read(), format="audio/mp3")

            st.success(thyroid_diagnosis)
            st.info(advice)

            # --- Summary Table ---
            normal_values = [35, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]

            summary_df = pd.DataFrame({
                'Metric': input_df.columns,
                'Your Value': input_data,
                'Normal Value': normal_values
            })
            st.subheader("📋 Your Thyroid Test Summary")
            st.table(summary_df)

            # --- Graph ---
            st.subheader("📊 Comparison Graph")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(normal_values, label='Normal', linestyle='--', marker='o', color='green')
            ax.plot(input_data, label='Your Input', linestyle='-', marker='x', color='red')
            ax.set_xticks(range(len(input_df.columns)))
            ax.set_xticklabels(input_df.columns, rotation=90)
            ax.set_ylabel('Values')
            ax.set_title('Your Input vs Normal Reference')
            ax.legend()
            st.pyplot(fig)

            # --- Care Tips ---
            st.subheader("🦋 Your Thyroid Care Tips")
            st.markdown(care)

            st.info("""
            ℹ️ **Note:** This prediction is for **educational/testing purposes only**.  
            Always consult a qualified doctor for medical advice.
            """)

            st.markdown("""
            ---
            ## 🦋 What is Thyroid Disease?

            Thyroid disease affects your thyroid gland, which controls your metabolism.  
            It can cause underactive (hypothyroidism) or overactive (hyperthyroidism) conditions.

            ✅ **How to Support Thyroid Health:**
            - Eat a balanced diet with enough iodine
            - Get your levels checked if you notice symptoms
            - Follow your doctor’s treatment plan
            """)
        except ValueError:
            st.error("⚠️ Please enter valid **numeric** values only!")
