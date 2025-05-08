import streamlit as st
import numpy as np
import joblib
from PIL import Image, UnidentifiedImageError
import os

# --- Streamlit config ---
st.set_page_config(page_title="Career Recommender", layout="wide")
st.title("Career Recommendation System for CS Students 🚀")
st.subheader("Fill your skill levels and get your ideal career!")

# --- Helper functions ---
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Model file not found: `{path}`")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
    return None

@st.cache_resource
def load_encoder(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"Encoder file not found: `{path}`")
    except Exception as e:
        st.error(f"Failed to load label encoder: {e}")
    return None

# --- Load model and encoder ---
model = load_model("career_recommendation_model.pkl")
label_encoder = load_encoder("label_encoder.pkl")

if model is None or label_encoder is None:
    st.stop()

# --- Skill mapping and columns ---
skill_mapping = {
    'Not Interested': 0,
    'Poor': 1,
    'Beginner': 2,
    'Average': 3,
    'Intermediate': 4,
    'Excellent': 5,
    'Professional': 6
}

skill_columns = [
    'Database Fundamentals', 'Computer Architecture', 'Distributed Computing Systems',
    'Cyber Security', 'Networking', 'Software Development', 'Programming Skills',
    'Project Management', 'Computer Forensics Fundamentals', 'Technical Communication',
    'AI ML', 'Software Engineering', 'Business Analysis', 'Communication skills',
    'Data Science', 'Troubleshooting skills', 'Graphics Designing'
]

# --- Career image and roadmap paths ---
career_images = {
    "Database Administrator": "database-admin.webp",
    "Hardware Engineer": "HwEngi.jpg",
    "Application Support Engineer": "ApplicationSupportEngineer.png",
    "Cyber Security Specialist": "CyberSecuritySpecialist.jpg",
    "Networking Engineer": "EnetWorkEngi.jpg",
    "Software Developer": "sw developer.webp",
    "API Specialist": "APISpecialist.jpg",
    "Project Manager": "ProjectManager.png",
    "Information Security Specialist": "InformationSecuritySpecialist.jpeg",
    "Technical Writer": "Technical Writer.jpg",
    "AI ML Specialist": "AI_ML_Specialist.jpeg",
    "Software Tester": "swtester.jpg",
    "Business Analyst": "Busnissanalyst.png",
    "Customer Service Executive": "CustomerServiceExecutive.jpg",
    "Data Scientist": "DataSciencist.png",
    "Helpdesk Engineer": "HelpDeskEnginner.webp",
    "Graphics Designer": "GraphicsDesinger.webp"
}

career_roadmaps = {
    "Database Administrator": "RDatabaseAdministrator.jpg",
    "Hardware Engineer": "RHardwareEngineer.jpg",
    "Application Support Engineer": "RApplicationSupportEngineer.jpg",
    "Cyber Security Specialist": "RCyberSecuritySpecialist.jpg",
    "Networking Engineer": "RetworkEngineer.jpg",
    "Software Developer": "RSoftwareDeveloper.jpg",
    "API Specialist": "RAPISpecialist.jpg",
    "Project Manager": "RProjectManagement.jpg",
    "Information Security Specialist": "RInformationSecuritySpecilaist.jpg",
    "Technical Writer": "RTechnicalWriter.jpg",
    "AI ML Specialist": "RAIMLSpecialist.jpg",
    "Software Tester": "RSoftwareTesting.jpg",
    "Business Analyst": "RBusinessAnalyst.jpg",
    "Customer Service Executive": "RCustomerServiceExecutive.jpg",
    "Data Scientist": "RDataScientist.jpg",
    "Helpdesk Engineer": "RHelpDeskEngineer.jpg",
    "Graphics Designer": "RGraphicDesigner.jpg"
}

# --- User input form ---
user_input = {}
with st.form("career_form"):
    for skill in skill_columns:
        user_input[skill] = st.selectbox(
            f"{skill}:",
            list(skill_mapping.keys()),
            index=list(skill_mapping.keys()).index("Beginner")
        )
    submitted = st.form_submit_button("Recommend Career")

# --- Prediction logic ---
if submitted:
    try:
        input_levels = [skill_mapping.get(user_input.get(skill, "Beginner"), 2) for skill in skill_columns]

        # --- Validation rules ---
        total_skills = len(input_levels)
        if all(level == 0 for level in input_levels):
            st.warning("⚠️ You selected 'Not Interested' for all skills. Please update your inputs to get a meaningful career recommendation.")
            st.stop()

        readiness_threshold = 0.70  # 70% of skills must be Beginner (2) or above
        ready_count = sum(level >= 2 for level in input_levels)
        if ready_count / total_skills < readiness_threshold:
            st.warning("⚠️ Most of your skills are marked 'Poor' or 'Not Interested'. Please review them to receive an accurate recommendation.")
            st.stop()

        # --- Predict ---
        input_data = np.array(input_levels).reshape(1, -1)

        if input_data.shape[1] != len(skill_columns):
            raise ValueError("Input shape mismatch with expected skill columns.")

        prediction = model.predict(input_data)
        predicted_role = label_encoder.inverse_transform(prediction)[0]
        st.success(f"🎯 Recommended Career Role: **{predicted_role}**")

        # --- Display image ---
        image_path = career_images.get(predicted_role)
        if image_path and os.path.exists(image_path):
            try:
                image = Image.open(image_path)
                st.image(image, caption=f"{predicted_role} Image", use_container_width=True)
            except UnidentifiedImageError:
                st.warning("⚠️ Image file is not a valid image format.")
        else:
            st.info("ℹ️ Image not available for this role.")

        # --- Display roadmap ---
        st.write(f"Here is a roadmap for becoming a {predicted_role}.")
        roadmap_path = career_roadmaps.get(predicted_role)
        if roadmap_path and os.path.exists(roadmap_path):
            try:
                roadmap = Image.open(roadmap_path)
                st.image(roadmap, caption=f"{predicted_role} Roadmap", use_container_width=True)
            except UnidentifiedImageError:
                st.warning("⚠️ Roadmap file is not a valid image format.")
        else:
            st.info("ℹ️ Roadmap not available for this role.")

    except ValueError as ve:
        st.error(f"❌ Input Error: {ve}")
    except Exception as e:
        st.error(f"❌ Unexpected error during prediction: {e}")
