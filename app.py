import streamlit as st
import tensorflow as tf
import joblib

model = tf.keras.models.load_model('student_performance.keras')
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title = "Student Performance Prediction",
    page_icon = "https://img.icons8.com/?size=100&id=21180&format=png&color=000000",
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

st.title('Student Performance Prediction 🎓')
tab1,tab2,tab3 = st.tabs(['Home','Prediction','About'])

st.markdown("""
<style>
.stApp{
     background: linear-gradient(135deg, #042F2E, #020617);
}
</style>
""",unsafe_allow_html=True)




    
with tab1:
    with st.container():
        st.markdown("""
        # 🎓 Student Performance Prediction

    This project uses an **Artificial Neural Network (ANN)** to predict student performance.

    The model uses factors such as:

    * 📚 Study Hours
    * 🏫 Attendance
    * 📝 Assignment Score
    * 📊 Previous Exam Score

    Enter the student's details to get the predicted performance.

    **🤖 Built with TensorFlow & Keras**


        """)

with tab2:
    with st.form("Details"):
        col1,col2 = st.columns(2)
        with col1:
           study_hours = st.number_input("Enter Study Hours",0)
           attendance_percent = st.number_input("Enter Attendance Percentage(without %)")
           assignment = st.number_input("Enter Overall Assignment Score")
           pre_exam = st.number_input("Enter Previous Exam Score") 
           sleep_hours = st.number_input("Enter Sleep Hours",0)

        with col2:
            screen_time = st.number_input("Enter Screentime",0)
            participation = st.number_input("Class Participation score out of 10",0)
            practice_test = st.number_input('Number Of practice Test Completed',0)
            extra = st.number_input("ExtraCurricular Hours Per Week")
            late = st.number_input("Number of Late Submissions",0)
            

        data = [[study_hours,attendance_percent,assignment,pre_exam,sleep_hours,
                 screen_time,participation,practice_test,extra,late]]
        input_data = scaler.transform(data)
        if st.form_submit_button("Predict"):
            if (study_hours == 0 or attendance_percent == 0 or assignment==0 or pre_exam==0 or sleep_hours==0 or screen_time==0 or participation==0 or practice_test== 0 or extra==0 or late ==0):
                st.error("Please Fill Required Fields!")
            else:
                prediction = model.predict(input_data)
                st.success(f"Predicted Final Exam Score: {prediction[0][0]:.2f}")
    st.markdown("""

    <div class='card'>
        <h3>Student Perormance Model</h3>
        <p>Model: Aritificial Neural Network</p>
    </div>
    <style>
    .card{
        background-color: blue;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid black;
        text-align: center
    }

    </style>
    """,unsafe_allow_html=True)
    
    st.write(" ")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div class="card1">
                <h3>Metric</h3>
                <p>MAE: 4.05</p>
            </div>
            <style>
            .card1{
                background-color: red;
                padding: 20px;
                border-radius: 15px;
                border: 1px solid black;
                box-shawdow: 0 4px 10px rgba(0,0,0,0.08)        
            }
            </style>
    """,unsafe_allow_html=True)

    st.write(" ")

    with col2:
        st.markdown("""
                <div class="card2">
                    <h3>Metric</h3>
                    <p>R² Score: 71 % </p>
                </div>
                <style>
                .card2{
                    background-color: red;
                    padding:20px;
                    border-radius: 15px;
                    border: 1px solid black;
                    
                }
                </style>
        """,unsafe_allow_html=True)
    st.write(" ")
    st.caption("Built by DevCodex")


with tab3:
        st.markdown("""
                ## 👨‍💻 About

This app is built as part of a learning and project journey in **Artificial Intelligence, Machine Learning, and Data Science**.

Built using **Python, TensorFlow, and Streamlit** with a focus on creating simple and practical ML applications.

### 🔗 Connect With Me

* **LinkedIn:** [https://www.linkedin.com/in/sudhasudalaimani/]
* **GitHub:** [https://github.com/thedevcodex]

Thanks for checking out this project! 

""")
        st.caption('Built by DevCodex')