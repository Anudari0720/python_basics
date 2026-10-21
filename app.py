import streamlit as st
import pandas as pd
import pickle

# 1. Хуудасны тохиргоо
st.set_page_config(page_title="EduPath AI", page_icon="🎓", layout="wide")

st.title("🎓 EduPath AI - Боловсрол & Карьер Зөвлөх Систем")
st.write("Таны сонирхол, зан төлөвт үндэслэн мэргэжил болон их сургууль санал болгох AI систем.")

# 2. Загвар болон Сургуулийн дата ачаалах
@st.cache_resource
def load_model():
    try:
        with open('edupath_model.pkl', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_universities():
    try:
        df = pd.read_csv('universities.csv')
        return df
    except:
        return pd.DataFrame({
            "Их сургууль": ["KAIST", "Seoul National University (SNU)", "Sungkyunkwan University (SKKU)", "МУИС", "ШУТИС"],
            "Улс": ["Өмнөд Солонгос", "Өмнөд Солонгос", "Өмнөд Солонгос", "Монгол", "Монгол"],
            "Сургалтын хөтөлбөр / Чиглэл": [
                "Computer Science, Brain & Cognitive Science, AI",
                "Computer Science, Engineering, Natural Sciences",
                "Software, Artificial Intelligence, HCI",
                "Мэдээллийн технологи, Компьютерийн ухаан, Биологи",
                "Програмгамгамжийн инженерчлэл, Мэдээллийн систем"
            ]
        })

model = load_model()
df_uni = load_universities()

# 3. Таб тохируулах (Асуулга, Чатбот, Их сургуулиуд)
tab1, tab2, tab3 = st.tabs(["📝 Мэргэжил тодорхойлох асуулга", "💬 AI Чатботтай ярилцах", "🏫 Их дээд сургуулиуд"])

# --- TAB 1: АСУУЛГА ---
with tab1:
    st.subheader("📋 Сурагчийн сонирхол ба зан төлөвийн асуулга")
    
    grade = st.selectbox("1. Одоо хэддүгээр анги вэ?", ["10-р анги", "11-р анги", "12-р анги"])
    
    q2 = st.radio(
        "2. Хэцүү бодлого, учрыг нь олдоггүй асуудал гараад ирвэл чиний тархи хамгийн эхэнд яаж ажилладаг вэ?",
        [
            "Компьютер, код бичих, математикийн хэцүү бодлого, дататай ажиллах талаас нь хардаг",
            "Хүний бие, анагаах ухаан, биологи, химийн лабораторийн туршилт талаас нь боддог",
            "Ном, интернетээс судалгаа хийж, шинжлэх ухааны үндэслэлтэй хариулт хайдаг",
            "Бусад хүмүүстэй ярилцаж, тэдний бодол санааг сонсож, сэтгэл зүйг нь ойлгохыг хичээдэг",
            "Бусдад тайлбарлаж өгөх эсвэл илтгэл тавих, дизайн гаргах, шинэ санаа олох арга хайдаг"
        ]
    )
    
    q3 = st.slider("3. Хэцүү логик таавар эсвэл кодтой таарахад сонирхол чинь хэр их хөдөлдөг вэ? (1-5)", 1.0, 5.0, 3.0, step=0.5)
    
    if st.button("🚀 Мэргэжил таамаглах"):
        st.markdown("---")
        if "Компьютер" in q2:
            st.success("🎓 Танд санал болгох чиглэл: **Computer Science / Artificial Intelligence**")
        elif "Хүний бие" in q2:
            st.success("🎓 Танд санал болгох чиглэл: **Biomedical Engineering / Brain & Cognitive Science**")
        elif "Бусад хүмүүстэй" in q2:
            st.success("🎓 Танд санал болгох чиглэл: **Cognitive Science / Affective Computing**")
        else:
            st.success("🎓 Танд санал болгох чиглэл: **Human-Computer Interaction (HCI) / Data Science**")

# --- TAB 2: ЧАТБОТ ---
with tab2:
    st.subheader("💬 AI Зөвлөхтэй чатлах")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Сайн байна уу! Би EduPath AI зөвлөх байна. Та өөрийн сонирхдог зүйл, хичээл эсвэл дуртай хоббигийнхоо талаар бичээрэй."}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Энд зурвасаа бичнэ үү (Жишээ нь: bi sags togloh durtai)..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        p_lower = prompt.lower()
        if any(w in p_lower for w in ["сагс", "sags", "спорт", "sport"]):
            response = "Спорт болон багийн ажиллагаанд сонирхолтой бол **Спортын аналитик, Кинезиологи (Хүний хөдөлгөөн судлал)** эсвэл **Спортын менежмент** чиглэлүүд маш тохиромжтой."
        elif any(w in p_lower for w in ["код", "компьютер", "ai", "магадлал", "it"]):
            response = "Технологид сонирхолтой бол **Computer Science, Software Engineering, AI** болон **Affective Computing** чиглэлүүд яг тохирно."
        elif any(w in p_lower for w in ["эмч", "анагаах", "биологи"]):
            response = "Байгалийн ухаанд сонирхолтой бол **Biomedical Engineering, Brain and Cognitive Science** эсвэл **Анагаах ухаан** чиглэлийг санал болгож байна."
        else:
            response = f"Таны бичсэн '{prompt}' гэсэн асуултыг шинжиж байна. Та дуртай хичээл эсвэл сурахыг хүсэж буй чиглэлээ арай тодорхой бичвэл би нарийн зөвлөгөө өгч чадна."

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --- TAB 3: ИХ СУРГУУЛИУД ---
with tab3:
    st.subheader("🏫 Санал болгох Их Дээд Сургуулиудын Жинхэнэ Дата")
    st.dataframe(df_uni, use_container_width=True)
