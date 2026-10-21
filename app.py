import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Хуудасны тохиргоо
st.set_page_config(page_title="EduPath AI - Карьер Зөвлөх", page_icon="🎓", layout="wide")

st.title("🎓 EduPath AI - Мэргэжил ба Их Сургууль Зөвлөх Систем")
st.write("Таны сонирхол, ур чадварт үндэслэн хамгийн тохиромжтой мэргэжлийг таамаглана.")

# 1. Загвар болон Дата ачаалах хэсэг
@st.cache_resource
def load_model():
    try:
        with open('edupath_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Модель ачаалахад алдаа гарлаа: {e}")
        return None

model = load_model()

# 2. Хэрэглэгчийн асуулга / Интерфэйс
st.subheader("📋 Сурагчийн асуулга")

col1, col2 = st.columns(2)

with col1:
    grade = st.selectbox("Та хэддүгээр анги вэ?", ["10-р анги", "11-р анги", "12-р анги"])
    interest_science = st.slider("Байгалийн ухаан, математикт сонирхолтой эсэх (1-5)", 1, 5, 3)
    interest_tech = st.slider("Компьютер, мэдээллийн технологид сонирхолтой эсэх (1-5)", 1, 5, 3)

with col2:
    interest_art = st.slider("Урлаг, дизайн, бүтээлч байдалд сонирхолтой эсэх (1-5)", 1, 5, 3)
    interest_social = st.slider("Нийгэм, хүмүүстэй харилцах чиглэлд сонирхолтой эсэх (1-5)", 1, 5, 3)

# 3. Мэргэжил таамаглах товч
if st.button("🚀 Мэргэжил таамаглах"):
    if model is not None:
        # Таамаглал хийх туршилтын логик
        st.success("🎉 Сонголтод үндэслэн шинжилж байна...")
        
        # Энд таны ML загварын таамаглах хэсэг ажиллана
        st.info("💡 Танд хамгийн тохиромжтой чиглэл: **Computer Science / Data Science**")
    else:
        st.warning("Модель файл хараахан ачаалагдаагүй байна. GitHub дээр edupath_model.pkl файл байгаа эсэхийг шалгаарай.")

# Их сургуулиудын жагсаалт шүүж харах
st.markdown("---")
st.subheader("🏫 Их дээд сургуулиудын мэдээлэл")
try:
    df_unis = pd.read_csv('data.csv - Form responses 1.csv')
    st.dataframe(df_unis.head())
except:
    st.write("Сургуулийн датаг ачаалж байна...")
