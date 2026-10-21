import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="EduPath AI", page_icon="🎓", layout="wide")

st.title("🎓 EduPath AI - Боловсрол & Карьер Зөвлөх Систем")
st.write("Сурагчдын зан төлөв, давуу тал болон сонирхолд үндэслэн тохирох мэргэжил, их сургуулийг санал болгох AI систем.")

# 1. ML Загвар ба Дата ачаалах
@st.cache_resource
def load_assets():
    model = None
    try:
        with open('edupath_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.warning(f"Модель файл ачаалахад анхааруулга: {e}")
    
    df_uni = None
    try:
        df_uni = pd.read_csv('data.csv - Form responses 1.csv')
    except:
        try:
            df_uni = pd.read_csv('data.csv')
        except:
            pass
            
    return model, df_uni

model, df_uni = load_assets()

# Session state тохируулах (Алхам алхмаар асуулт асуух)
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'answers' not in st.session_state:
    st.session_state.answers = {}

st.markdown("---")

# --- 1-р алхам: Үндсэн асуулга ---
if st.session_state.step == 1:
    st.subheader("📝 Алхам 1/2: Сурагчийн зан төлөв ба суралцах арга барил")
    
    grade = st.selectbox("1. Одоо хэддүгээр анги вэ?", ["10-р анги", "11-р анги", "12-р анги"])
    
    q2 = st.radio(
        "2. Хэцүү бодлого, учрыг нь олдоггүй асуудал гараад ирвэл чиний тархи хамгийн эхэнд яаж ажилладаг вэ?",
        [
            "Ном, интернетээс судалгаа хийж, шинжлэх ухааны үндэслэлтэй хариулт хайдаг",
            "Хүний бие, анагаах ухаан, биологи, химийн лабораторийн туршилт талаас нь боддог",
            "Компьютер, код бичих, математикийн хэцүү бодлого, дататай ажиллах талаас нь хардаг",
            "Бусад хүмүүстэй ярилцаж, тэдний бодол санааг сонсож, сэтгэл зүйг нь ойлгохыг хичээдэг",
            "Бусдад тайлбарлаж өгөх эсвэл илтгэл тавих, дизайн гаргах, шинэ санаа олох арга хайдаг"
        ]
    )
    
    q3 = st.radio(
        "3. Ямар хичээл эсвэл зүйл дээр суухад цаг хугацаа хурдан өнгөрчихсөн байдаг вэ?",
        [
            "Компьютер, код бичих, код бичиж турших эсвэл логик тоглоом тоглох",
            "Хуний бие, анагаах ухаан, биологи, химийн лабораторийн туршилт хийх",
            "Ном, интернетээс судалгаа хийх, эссэ бичих, түүх, нийгмийн хичээл судалж унших",
            "Зураг зурах, дизайн гаргах, бичлэг эвлүүлэх, уран бүтээл хийх",
            "Спорт, сагсан бөмбөг, шатар тоглох, гадаад хэл үзэх"
        ]
    )

    if st.button("Дараагийн алхам ➔"):
        st.session_state.answers['grade'] = grade
        st.session_state.answers['q2'] = q2
        st.session_state.answers['q3'] = q3
        st.session_state.step = 2
        st.rerun()

# --- 2-р алхам: Өөрөө өөрийгөө үнэлэх ---
elif st.session_state.step == 2:
    st.subheader("📊 Алхам 2/2: Зан чанар ба Үнэлгээ")
    
    q4 = st.slider("4. Хэцүү логик таавар эсвэл кодтой таарахад сонирхол чинь хэр их хөдөлдөг вэ? (1-5)", 1, 5, 3)
    q5 = st.slider("5. Хийж байсан зүйл чинь олон дахин алдаа гарвал яадаг вэ? (1-5)", 1, 5, 3)
    q6 = st.slider("6. Дэлгэц ширтсэний өдөрт ганцаараа 3-4 цаг тасралтгүй сууж ажиллах ямар санагддаг вэ? (1-5)", 1.0, 5.0, 4.0, step=0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Буцах"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("🚀 Мэргэжил таамаглах"):
            st.session_state.answers['q4'] = q4
            st.session_state.answers['q5'] = q5
            st.session_state.answers['q6'] = q6
            st.session_state.step = 3
            st.rerun()

# --- 3-р алхам: Шинжилгээ ба Үр дүн ---
elif st.session_state.step == 3:
    st.subheader("🎯 Шинжилгээний Дүн ба Таамаглал")
    
    # Сонголтоос хамаарч мэргэжил тодорхойлох логик
    q2_ans = st.session_state.answers.get('q2', '')
    q3_ans = st.session_state.answers.get('q3', '')
    
    predicted_major = "Computer Science / AI Engineering"
    if "анагаах" in q2_ans.lower() or "биологи" in q3_ans.lower():
        predicted_major = "Biomedical Engineering / Brain & Cognitive Science"
    elif "дизайн" in q2_ans.lower() or "зураг" in q3_ans.lower():
        predicted_major = "Human-Computer Interaction (HCI) / UX-UI Design"
    elif "нийгэм" in q2_ans.lower() or "сэтгэл зүй" in q2_ans.lower():
        predicted_major = "Cognitive Science / Affective Computing"
        
    st.success(f"🎓 Таны хариултад үндэслэсэн тохиромжтой чиглэл: **{predicted_major}**")
    
    st.markdown("---")
    st.subheader("🏫 Санал болгох Их Дээд Сургуулиуд ба Дата")
    
    if df_uni is not None:
        st.dataframe(df_uni)
    else:
        st.info("Сурагчдын болон сургуулийн асуулгын дататай ажиллаж байна.")
        
    if st.button("🔄 Дахин асуулга өгөх"):
        st.session_state.step = 1
        st.session_state.answers = {}
        st.rerun()
    import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="EduPath AI Chatbot", page_icon="🎓", layout="centered")

st.title("🎓 EduPath AI - Карьер & Сургууль Зөвлөх Чатбот")
st.write("Сайн байна уу? Би таны боловсрол болон карьер сонголтод туслах AI чатбот байна.")

# Загвар болон дата ачаалах
@st.cache_resource
def load_assets():
    model = None
    try:
        with open('edupath_model.pkl', 'rb') as f:
            model = pickle.load(f)
    except:
        pass
    
    df_uni = None
    try:
        df_uni = pd.read_csv('data.csv - Form responses 1.csv')
    except:
        try:
            df_uni = pd.read_csv('data.csv')
        except:
            pass
    return model, df_uni

model, df_uni = load_assets()

# Чат болон мессежийн түүх хадгалах state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Сайн байна уу! Та ямар мэргэжил сонирхож байна вэ? Эсвэл өөрийнхөө сонирхлын талаар хэлээрэй."}
    ]

# Өмнөх мессежүүдийг дэлгэцэнд харуулах
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Хэрэглэгч чат руу юм бичих хэсэг (Chat Input)
if prompt := st.chat_input("Энд зурвасаа бичнэ үү (Жишээ нь: Computer Science ямар вэ?)..."):
    # Хэрэглэгчийн бичсэнийг чатад нэмэх
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Чатботын хариу өгөх хэсэг
    response = f"Таны асуусан '{prompt}' гэсэн сэдвээр судлаад үзэхэд Computer Science, Artificial Intelligence болон Affective Computing чиглэлүүд маш тохиромжтой харагдаж байна! Мөн манай датабааз дээр холбогдох их сургуулиуд бүртгэгдсэн байгаа."
    
    if "сургууль" in prompt.lower() or "universities" in prompt.lower():
        response = "Манай систем дэх их сургуулиудын жагсаалтыг доорх дата хүснэгтээс хараарай!"
        if df_uni is not None:
            st.dataframe(df_uni.head())

    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
