import datetime
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="EduPath AI - Smart Student Guidance & Career Platform",
    page_icon="🎓",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; 
        border-radius: 8px 8px 0px 0px; 
        padding: 10px 20px; 
        font-weight: 600;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stTabs [aria-selected="true"] { 
        background-color: #4f46e5 !important; 
        color: white !important; 
    }
    .prediction-box {
        background-color: #eef2ff;
        border: 2px solid #6366f1;
        padding: 20px;
        border-radius: 12px;
        margin-top: 15px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. DATA LOADING & ML MODEL TRAINING (Cached)
# ==========================================
@st.cache_data
def load_data_and_train_model():
  try:
    df_eysh = pd.read_csv("Eysh_predictor_data - Sheet1.csv")
    df_eysh["program_name"] = df_eysh["program_name"].astype(str).str.strip()
    df_eysh["university_name"] = (
        df_eysh["university_name"].astype(str).str.strip()
    )
    df_eysh["exam_subject_1"] = (
        df_eysh["exam_subject_1"].astype(str).str.strip()
    )
  except Exception:
    df_eysh = pd.DataFrame()

  try:
    df_survey = pd.read_csv("data.csv - Form responses 1.csv")
  except Exception:
    df_survey = pd.DataFrame()

  # Machine Learning Model Training setup
  model = None
  encoders = {}
  target_encoder = None

  if not df_survey.empty:
    try:
      # Clean survey data for ML
      df_ml = df_survey.dropna().copy()
      cols = df_ml.columns

      # Features selection
      feature_cols = [
          cols[2],
          cols[3],
          cols[4],
          cols[8],
          cols[9],
          cols[13],
      ]  # problem-solving, interest, logic score, stress, IT speed, fav subject
      target_col = cols[14]  # Preferred sector

      X = df_ml[feature_cols].copy()
      y = df_ml[target_col].copy()

      # Encode categorical features
      for col in feature_cols:
        if X[col].dtype == "object":
          le = LabelEncoder()
          X[col] = le.fit_transform(X[col].astype(str))
          encoders[col] = le

      # Encode target
      target_encoder = LabelEncoder()
      y_encoded = target_encoder.fit_transform(y.astype(str))

      # Train Random Forest Model
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X, y_encoded)
    except Exception as e:
      print("ML Training Exception:", e)

  return df_eysh, df_survey, model, encoders, target_encoder


df_eysh, df_survey, ml_model, encoders, target_encoder = (
    load_data_and_train_model()
)


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "stress_result" not in st.session_state:
  st.session_state.stress_result = None
if "goal_data" not in st.session_state:
  st.session_state.goal_data = {
      "goal_name": "ЭЕШ болон Их сургуулиа амжилттай сонгох",
      "duration": "4 долоо хоног",
  }
if "checklist" not in st.session_state:
  st.session_state.checklist = {
      "Өдөр бүр 30 мин ЭЕШ-ын бэлтгэл / тест ажиллах": False,
      "Гадаад хэл болон судалж буй хичээлийн давтлага хийх": False,
      "7-8 цаг тогтмол унтаж амарөх": False,
      "15 минут биеийн тамир / дасгал хөдөлгөөн хийх": False,
  }
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
      {
          "role": "assistant",
          "content": (
              "Сайн байна уу? Би таны **EduPath AI** ухаалаг залуусын зөвлөх"
              " байна. Сургагдсан ML модел болон бодит датанд суурилан танд"
              " туслахад бэлэн байна!"
          ),
      }
  ]

# ==========================================
# 4. SIDEBAR & HEADER
# ==========================================
st.sidebar.image(
    "https://img.icons8.com/color/96/graduation-cap.png", width=80
)
st.sidebar.title("EduPath AI 🚀")
st.sidebar.markdown(
    "**Ахлах ангийн сурагчдын сэтгэл зүй, олон талт мэргэжил сонголт болон ЭЕШ"
    " төлөвлөлтийн ухаалаг платформ.**"
)
st.sidebar.markdown("---")

if ml_model is not None:
  st.sidebar.success("🤖 **ML Model:** Random Forest Classifier сургагдсан!")
if not df_survey.empty and not df_eysh.empty:
  st.sidebar.info(
      f"📊 **Сургалтын дата:**\n- Асуулга өгсөн сурагчид: {len(df_survey)}\n-"
      f" Боломжит хөтөлбөр: {len(df_eysh)}"
  )

st.title("🎓 EduPath AI: Ухаалаг Сурагчийн Зөвлөх Платформ")
st.markdown(
    "Сэтгэл зүйгээ үнэлж, **Machine Learning** загвараар өөрт тохирох"
    " салбараа таамаглан, бодит ЭЕШ-ын дататай танилцаарай."
)

# ==========================================
# 5. MAIN TABS STRUCTURE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Стресс ба Ачааллын Үнэлгээ",
    "🎯 Зорилго & 4 Долоо Хоногийн Төлөвлөгч",
    "🤖 ML AI Мэргэжил Таамаглагч & Сургууль Хайгч",
    "💬 AI Сэтгэл Зүйч & Карьер Чатбот",
    "📥 Карьерын Тайлан Татах",
])

# ------------------------------------------
# TAB 1: STRESS & BURNOUT DIAGNOSTIC
# ------------------------------------------
with tab1:
  st.header("📊 Сэтгэл Зүй ба Хичээлийн Ачааллын Үнэлгээ")
  col1, col2 = st.columns(2)
  with col1:
    study_hours = st.slider(
        "Өдөрт хичээл хийх / давтах дундаж цаг:", 1, 14, 6
    )
    sleep_hours = st.slider("Өдөрт унтаж буй дундаж цаг:", 3, 10, 7)
  with col2:
    exam_stress = st.selectbox(
        "Шалгалт болон ЭЕШ-ын стрессийн түвшин:",
        ["Бага", "Дунд", "Өндөр", "Маш өндөр"],
    )
    activity_level = st.selectbox(
        "Биеийн идэвх болон амралтын байдал:",
        [
            "Тогтмол хөдөлгөөн сайтай",
            "Заримдаа амардаг",
            "Байнга суугаа байрлалтай / ядардаг",
        ],
    )

  if st.button("Стрессийн түвшинг тооцоолох", type="primary"):
    score = 0
    if study_hours > 9:
      score += 2
    if sleep_hours < 6:
      score += 3
    elif sleep_hours < 7:
      score += 1
    if exam_stress in ["Өндөр", "Маш өндөр"]:
      score += 3
    if activity_level == "Байнга суугаа байрлалтай / ядардаг":
      score += 2

    if score <= 2:
      st.session_state.stress_result = "Бага (Тэнцвэртэй)"
      st.success(
          "🎉 **Стрессийн түвшин: Бага.** Таны бие махбодь болон сэтгэл зүй"
          " маш сайн тэнцвэртэй байна."
      )
    elif score <= 5:
      st.session_state.stress_result = "Дунд зэрэг (Анхаарах)"
      st.warning(
          "⚠️ **Стрессийн түвшин: Дунд зэрэг.** Өдөрт 15-20 минут салхинд гарч,"
          " унтаж хэвшээрэй."
      )
    else:
      st.session_state.stress_result = "Өндөр (Анхаарал татах)"
      st.error("🚨 **Стрессийн түвшин: Өндөр!** Сайн амарч, зөвлөгөө аваарай.")

# ------------------------------------------
# TAB 2: GOAL & ROADMAP PLANNER
# ------------------------------------------
with tab2:
  st.header("🎯 Зорилго ба 4 Долоо Хоногийн Төлөвлөгч")
  goal_choice = st.selectbox(
      "Үндсэн зорилгоо сонгоно уу:",
      [
          "ЭЕШ-ын оноогоо ахиулах & Сургуулиа сонгох",
          "Анагаах / Сэтгэл зүй / Нийгмийн чиглэлээр бэлдэх",
          "Бизнес, Эдийн засаг & Энтрепренершип төсөл хэрэгжүүлэх",
          "Компьютер, IT & AI Технологийн төсөл дуусгах",
          "IELTS / SAT оноогоо ахиулж, Гадаадын сургуульд бэлдэх",
      ],
  )
  st.session_state.goal_data["goal_name"] = goal_choice

  col_w1, col_w2, col_w3, col_w4 = st.columns(4)
  with col_w1:
    st.markdown("**1-р долоо хоног**\n* Судалгаа ба суралцах ажилдаа бэлдэх")
  with col_w2:
    st.markdown("**2-р долоо хоног**\n* Давтлага хийж, даалгавар ажиллах")
  with col_w3:
    st.markdown("**3-р долоо хоног**\n* Шалгалтын сорилт өгөх & Алдаа засах")
  with col_w4:
    st.markdown("**4-р долоо хоног**\n* Эцсийн байдлаар дүгнэх & Хүсэлт гаргах")

  st.markdown("---")
  st.subheader("✅ Өнөөдрийн Хийх Дадлууд (Checklist)")
  completed_count = sum([
      st.checkbox(k, value=v) for k, v in st.session_state.checklist.items()
  ])
  st.progress(completed_count / len(st.session_state.checklist))

# ------------------------------------------
# TAB 3: MACHINE LEARNING CAREER PREDICTOR & EYSH SEARCH
# ------------------------------------------
with tab3:
  st.header("🤖 Machine Learning Санал Болгох Систем & Сургууль Хайгч")
  st.markdown(
      "Судалгааны 48 сурагчийн дата дээр сургасан **Random Forest ML** модель"
      " таны хариултад тулгуурлан хамгийн тохиромжтой салбарыг таамаглана."
  )

  if ml_model is not None and not df_survey.empty:
    st.subheader("🧠 1. Өөрийн профайлыг оруулж ML санал болголт авах:")
    cols_s = df_survey.columns

    col_m1, col_m2 = st.columns(2)
    with col_m1:
      input_q1 = st.selectbox(
          "Хэцүү асуудал гарахад таны тархи яаж ажилладаг вэ?",
          df_survey[cols_s[2]].dropna().unique(),
      )
      input_q2 = st.selectbox(
          "Ямар зүйл дээр суухад цаг хурдан өнгөрдөг вэ?",
          df_survey[cols_s[3]].dropna().unique(),
      )
      input_q3 = st.slider(
          "Логик болон кодод сонирхолтой байдал (1-5):", 1, 5, 3
      )
    with col_m2:
      input_q4 = st.slider("Стресс даах чадвар (1-5):", 1, 5, 3)
      input_q5 = st.slider("Шинэ IT / AI хэрэгсэл сурах хурд (1-5):", 1, 5, 3)
      input_q6 = st.selectbox(
          "Дуртай бөгөөд ахицтай хичээлийн чиглэл:",
          df_survey[cols_s[13]].dropna().unique(),
      )

    if st.button("🤖 ML Моделиор Таамаглах", type="primary"):
      try:
        # Encode inputs
        val_q1 = (
            encoders[cols_s[2]].transform([input_q1])[0]
            if input_q1 in encoders[cols_s[2]].classes_
            else 0
        )
        val_q2 = (
            encoders[cols_s[3]].transform([input_q2])[0]
            if input_q2 in encoders[cols_s[3]].classes_
            else 0
        )
        val_q6 = (
            encoders[cols_s[13]].transform([input_q6])[0]
            if input_q6 in encoders[cols_s[13]].classes_
            else 0
        )

        features = [[val_q1, val_q2, input_q3, input_q4, input_q5, val_q6]]
        pred_class = ml_model.predict(features)[0]
        predicted_sector = target_encoder.inverse_transform([pred_class])[0]

        st.markdown(
            f"""
            <div class="prediction-box">
                <h3>🎉 AI Таамагласан Тохиромжтой Салбар:</h3>
                <h2>🌟 {predicted_sector}</h2>
                <p>Энэхүү таамаглалыг асуулга өгсөн бусад сурагчдын сэтгэл зүй ба сонирхлын загвар (Random Forest) дээр үндэслэн гаргалаа.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
      except Exception as e:
        st.error(f"Таамаглахад алдаа гарлаа: {e}")

  st.markdown("---")
  st.subheader("🔍 2. ЭЕШ-ын Сургууль ба Хөтөлбөр Хайлтын Систем (Бодит дата)")

  if not df_eysh.empty:
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
      unis = ["Бүгд"] + sorted(df_eysh["university_name"].unique().tolist())
      selected_uni = st.selectbox("Их Сургууль сонгох:", unis)
    with col_f2:
      subjs = ["Бүгд"] + sorted(df_eysh["exam_subject_1"].unique().tolist())
      selected_subj = st.selectbox("ЭЕШ-ын 1-р хичээл:", subjs)
    with col_f3:
      search_keyword = st.text_input("Мэргэжлийн нэрээр хайх:", "")

    filtered_df = df_eysh.copy()
    if selected_uni != "Бүгд":
      filtered_df = filtered_df[filtered_df["university_name"] == selected_uni]
    if selected_subj != "Бүгд":
      filtered_df = filtered_df[filtered_df["exam_subject_1"] == selected_subj]
    if search_keyword:
      filtered_df = filtered_df[
          filtered_df["program_name"].str.contains(search_keyword, case=False)
      ]

    st.markdown(f"**Олдсон хөтөлбөрийн тоо:** {len(filtered_df)}")
    st.dataframe(
        filtered_df[[
            "university_name",
            "program_name",
            "exam_subject_1",
            "threshold_score_1",
            "exam_subject_2",
            "threshold_score_2",
            "scholar_name",
        ]],
        use_container_width=True,
    )

# ------------------------------------------
# TAB 4: AI CHATBOT COUNSELOR
# ------------------------------------------
with tab4:
  st.header("💬 AI Сэтгэл Зүйч & Карьер Зөвлөх Чатбот")
  for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if user_input := st.chat_input("Энд асуултаа бичнэ үү:"):
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
      st.markdown(user_input)

    query_lower = user_input.lower()
    if "стресс" in query_lower or "ядар" in query_lower:
      bot_response = (
          "Стресс болон ядралт мэдрэгдэж байвал та түр завсарлаад гүнзгий"
          " амьсгаа аваарай."
      )
    elif "эеш" in query_lower or "сургууль" in query_lower:
      bot_response = (
          "Tab 3 дээр байгаа ML Модел болон ЭЕШ-ын датаг ашиглан сургуулиа"
          " шүүж хараарай."
      )
    else:
      bot_response = (
          f"Таны асуултыг ойлголоо: '{user_input}'. EduPath AI платформ танд"
          " туслахад бэлэн байна."
      )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": bot_response}
    )
    with st.chat_message("assistant"):
      st.markdown(bot_response)

# ------------------------------------------
# TAB 5: EXPORT REPORT
# ------------------------------------------
with tab5:
  st.header("📥 Карьер ба Зорилгын Тайлан Татах")
  report_text = f"""========================================
EDUPATH AI - СУРАГЧИЙН УХААЛА格 ТАЙЛАН
Үүсгэсэн огноо: {datetime.date.today()}
========================================
- Стрессийн үнэлгээ: {st.session_state.stress_result}
- Сонгосон зорилго: {st.session_state.goal_data['goal_name']}
========================================
"""
  st.text_area("Тайлангийн текст:", report_text, height=200)
  st.download_button(
      label="📥 Файл татах (.txt)",
      data=report_text,
      file_name="EduPath_AI_Report.txt",
      mime="text/plain",
  )

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>EduPath AI © 2026 | Capstone"
    " Project</p>",
    unsafe_allow_html=True,
)
