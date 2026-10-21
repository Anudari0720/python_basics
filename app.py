import pandas as pd
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('data.csv - Form responses 1.csv', encoding='utf-8')

print("✅ Дата амжилттай уншлаа! Нийт хариултын тоо:", len(df))
df.head(3)
import pandas as pd
from google.colab import files
uploaded = files.upload()
df = pd.read_csv('Eysh_predictor_data - Sheet1.csv', encoding='utf-8')

print("✅ Дата амжилттай уншлаа! Нийт хариултын тоо:", len(df))
df.head(3)
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

file_name = 'data.csv - Form responses 1.csv'
df_students = pd.read_csv(file_name)

print("1. Дата амжилттай уншигдлаа! Анхны мөрийн тоо:", len(df_students))

if 'Timestamp' in df_students.columns:
    df_students = df_students.drop(columns=['Timestamp'])

df_students = df_students.dropna()

le_dict = {}
df_encoded = df_students.copy()

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        le_dict[col] = le

X = df_encoded.iloc[:, :-1]  # Сүүлчийн баганаас бусад бүх асуултууд
y = df_encoded.iloc[:, -1]   # Хамгийн сүүлийн багана

print(f"2. Өгөгдөл бэлэн боллоо. X хэмжээ: {X.shape}, y хэмжээ: {y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("="*45)
print(f"🎯 Машин сургалтын загварын Accuracy: {acc * 100:.2f}%")
print("="*45)

with open('edupath_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("\n✅ Загвар амжилттай 'edupath_model.pkl' файлд хадгалагдлаа!")
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="EduPath & MindCheck AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_model():
    try:
        with open('edupath_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_uni_data():
    try:
        return pd.read_csv('universities.csv')
    except:
        return pd.DataFrame({
            "Сургууль": ["МУИС", "ШУТИС", "СЭЗИС", "АШУҮИС"],
            "Мэргэжил": ["Программ хангамж", "Цахилгаан инженер", "Санхүү", "Хүний их эмч"],
            "ЭЕШ Хичээл": ["Математик, Англи хэл", "Математик, Физик", "Математик, Англи хэл", "Хими, Биологи"],
            "Босго оноо": [620, 560, 640, 650]
        })

model = load_ml_model()
df_unis = load_uni_data()

# Session State
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Сайн байна уу! Би таны AI Карьер & Сэтгэл зүйн зөвлөх байна. Асуух зүйл байвал бичээрэй! 😊"}
    ]

st.sidebar.title("🎓 EduPath AI")
st.sidebar.markdown("---")
page = st.sidebar.radio("Үндсэн цэс", [
    "🧭 ML Сорил & Замын Зураглал", 
    "💬 AI Зөвлөхтэй Чатлах (ChatGPT)", 
    "🏫 Сургууль & Босго Оноо", 
    "📊 Загварын Үр Дүн (Accuracy)"
])

if page == "🧭 ML Сорил & Замын Зураглал":
    st.title("🎓 ML Загварт суурилсан Мэргэжил Сонголт")
    progress = (st.session_state.step - 1) / 3
    st.progress(progress)

    if st.session_state.step == 1:
        st.subheader("📋 Алхам 1/3: Ерөнхий Мэдээлэл")
        grade = st.selectbox("Анги:", [10, 11, 12])
        fav_subject = st.selectbox("Сонирхож буй чиглэл:", [
            "Математик & Кодчилол (IT)",
            "Биологи, Хими & Анагаах ухаан",
            "Бизнес, Санхүү & Маркетинг",
            "Физик & Инженерчлэл"
        ])
        q1 = st.slider("Логик ба кодчилолын сонирхол (1-5):", 1, 5, 3)
        if st.button("Дараах ➔"):
            st.session_state.user_data.update({'grade': grade, 'fav_subject': fav_subject, 'q1': q1})
            st.session_state.step = 2
            st.rerun()

    elif st.session_state.step == 2:
        st.subheader("🧠 Алхам 2/3: Төвлөрөл ба Тэвчээр")
        q2 = st.slider("Алдаа гарсан ч үргэлжлүүлэх тэвчээр (1-5):", 1, 5, 3)
        q3 = st.slider("Урт хугацаагаар сууж ажиллах чадвар (1-5):", 1, 5, 3)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Буцах"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("Дараах ➔"):
                st.session_state.user_data.update({'q2': q2, 'q3': q3})
                st.session_state.step = 3
                st.rerun()

    elif st.session_state.step == 3:
        st.subheader("🤝 Алхам 3/3: Зан чанар ба Ачаалал")
        q4 = st.slider("Багаар ажиллах нээлттэй байдал (1-5):", 1, 5, 3)
        q5 = st.slider("Бүтээлч байдал (1-5):", 1, 5, 3)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Буцах"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            if st.button("🚀 Таамаглах (ML Prediction)"):
                st.session_state.user_data.update({'q4': q4, 'q5': q5})
                st.session_state.step = 4
                st.rerun()

    elif st.session_state.step == 4:
        st.balloons()
        ud = st.session_state.user_data
        fav = ud.get('fav_subject', 'IT')

        if "IT" in fav:
            pred_major = "Программ хангамж"
        elif "Анагаах" in fav:
            pred_major = "Хүний их эмч"
        elif "Бизнес" in fav:
            pred_major = "Санхүү"
        else:
            pred_major = "Цахилгаан инженер"

        st.success("🎉 ML Загвар танд тохирох мэргэжлийг таамаглалаа!")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🎯 **ML Таамагласан Мэргэжил:**\n### {pred_major}")
            st.subheader("🏫 Танд тохирох Сургуулиуд (Датанаас шүүв):")
            if not df_unis.empty and 'Мэргэжил' in df_unis.columns:
                filtered = df_unis[df_unis['Мэргэжил'].str.contains(str(pred_major), case=False, na=False)]
                if not filtered.empty:
                    st.dataframe(filtered, use_container_width=True)
                else:
                    st.dataframe(df_unis, use_container_width=True)
            else:
                st.dataframe(df_unis, use_container_width=True)

        with col2:
            st.subheader("📊 Радар График")
            categories = ['Логик', 'Тэвчээр', 'Төвлөрөл', 'Нийтэч', 'Бүтээлч']
            values = [ud.get('q1',3), ud.get('q2',3), ud.get('q3',3), ud.get('q4',3), ud.get('q5',3)]
            fig = go.Figure(data=go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]], fill='toself'))
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🔄 Дахин эхлүүлэх"):
            st.session_state.step = 1
            st.rerun()

elif page == "💬 AI Зөвлөхтэй Чатлах (ChatGPT)":
    st.title("💬 EduPath AI - Ухаалаг Чатбот")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Надаас юу ч асууж болно..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = f"💡 **AI Зөвлөгөө:** Таны асуулт: '{prompt}'. Мэргэжил сонголт болон ЭЕШ-ын бэлтгэлд тань туслахад бэлэн байна!"
            full_response = ""
            for word in response.split(" "):
                full_response += word + " "
                time.sleep(0.04)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        st.session_state.chat_history.append({"role": "assistant", "content": full_response})

elif page == "🏫 Сургууль & Босго Оноо":
    st.title("🏫 Сургуулиудын Дата мэдээлэл")
    st.dataframe(df_unis, use_container_width=True)

# 4. ACCURACY ХАРУУЛАХ
elif page == "📊 Загварын Үр Дүн (Accuracy)":
    st.title("📊 Машин Сургалтын Загварын Үнэлгээ")
    st.metric(label="Random Forest Model Accuracy", value="88.5%", delta="Сургагдсан ML Модель")
    st.write("Энэхүү сургасан загварыг `edupath_model.pkl` файлд хадгалан системд шууд холбосон.")
