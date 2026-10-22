import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import streamlit as st

st.set_page_config(
    page_title="EduPath AI - Student Career Guidance",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 EduPath AI: Сурагчийн Зөвлөх Платформ")

@st.cache_data
def load_data_and_train_ml():
  try:
    df_eysh = pd.read_csv("Eysh_predictor_data - Sheet1.csv")
    df_eysh["program_name"] = df_eysh["program_name"].astype(str).str.strip()
    df_eysh["university_name"] = (
        df_eysh["university_name"].astype(str).str.strip()
    )
  except Exception:
    df_eysh = pd.DataFrame()

  try:
    df_survey = pd.read_csv("data.csv - Form responses 1.csv")
  except Exception:
    df_survey = pd.DataFrame()

  model = None
  encoders = {}
  target_encoder = None

  # --- Machine Learning Classification Model ---
  if not df_survey.empty:
    try:
      cols = df_survey.columns
      # Key Feature Columns from Survey
      feature_cols = [cols[2], cols[3], cols[4], cols[8], cols[9], cols[13]]
      target_col = cols[14]  # Ирээдүйд сонирхож буй салбар

      df_ml = df_survey[feature_cols + [target_col]].dropna().copy()

      X = df_ml[feature_cols].copy()
      y = df_ml[target_col].copy()

      # Encoding categorical variables
      for col in feature_cols:
        if X[col].dtype == "object":
          le = LabelEncoder()
          X[col] = le.fit_transform(X[col].astype(str))
          encoders[col] = le

      target_encoder = LabelEncoder()
      y_encoded = target_encoder.fit_transform(y.astype(str))

      # Train ML Random Forest Classifier Model
      model = RandomForestClassifier(n_estimators=100, random_state=42)
      model.fit(X, y_encoded)
    except Exception as e:
      print("ML Training error:", e)

  return df_eysh, df_survey, model, encoders, target_encoder


df_eysh, df_survey, ml_model, encoders, target_encoder = (
    load_data_and_train_ml()
)

# Session state setup
if "stress_score" not in st.session_state:
  st.session_state.stress_score = "Үнэлгээ хийгдээгүй"

if "checklist" not in st.session_state:
  st.session_state.checklist = {
      "Өдөр бүр 25 мин Python код бичих": False,
      "Өдөрт 10 шинэ Англи үг цээжлэх": False,
      "7-8 цаг тогтмол унтаж амрах": False,
  }

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 1. Стресс Үнэлгээ",
    "🎯 2. Зорилго ба Төлөвлөгөө",
    "🤖 3. Мэргэжил Таамаглагч & Сургууль Хайгч",
    "💬 4. AI Чатбот",
])

with tab1:
  st.header("📊 Сэтгэл Зүй ба Стрессийн Оношлогоо")
  col1, col2 = st.columns(2)
  with col1:
    study = st.slider("Өдөрт хичээл хийдэг цаг:", 1, 12, 6)
    sleep = st.slider("Өдөрт унтдаг цаг:", 3, 10, 7)
  with col2:
    exam = st.selectbox(
        "Шалгалтын стрессийн түвшин:",
        ["Бага (Low)", "Өндөр (High)", "Маш өндөр"],
    )

  if st.button("Стресс тооцоолох", type="primary"):
    if sleep < 6 or study > 9 or exam == "Маш өндөр":
      st.session_state.stress_score = "Өндөр Стресстэй 🚨"
      st.error(
          "Таны стрессийн түвшин өндөр байна! Сайн амарч, хичээлийн цагаа"
          " багасгаарай."
      )
    else:
      st.session_state.stress_score = "Хэвийн / Бага Стресс 🟢"
      st.success("Таны бие ба сэтгэл зүй тэнцвэртэй байна!")

with tab2:
  st.header("🎯 Зорилго ба 4 Долоо Хоногийн Стратеги")
  goal = st.selectbox(
      "Үндсэн зорилгоо сонгоно уу:",
      [
          "SAT оноогоо ахиулах",
          "IELTS оноогоо ахиулах",
          "ЭЕШ болон Их сургуулийн бэлтгэл хийх",
      ],
  )
  st.write(f"**Сонгосон зорилго:** {goal}")

  st.subheader("📅 4 Долоо Хоногийн Төлөвлөгөө")
  c1, c2, c3, c4 = st.columns(4)
  c1.info("1-р долоо хоног: Судалгаа хийх")
  c2.info("2-р долоо хоног: Давтлага хийх")
  c3.info("3-р долоо хоног: Төсөл/Дасгал хийх")
  c4.info("4-р долоо хоног: Дүгнэх ба Шалгах")

  st.subheader("✅ Өдөр Тутам Хийх Дадлууд")
  completed = 0
  for task in list(st.session_state.checklist.keys()):
    val = st.checkbox(task, value=st.session_state.checklist[task])
    st.session_state.checklist[task] = val
    if val:
      completed += 1
  st.progress(completed / len(st.session_state.checklist))
with tab3:
  st.header(" Санал Болгох Систем")
  
  if ml_model is not None and not df_survey.empty:
    cols_s = df_survey.columns
    st.subheader("📝 Сурагчийн профайл оруулах:")

    col_m1, col_m2 = st.columns(2)
    with col_m1:
      q1 = st.selectbox(
          "Хэцүү асуудал гарахад тархи яаж ажилладаг вэ?:",
          df_survey[cols_s[2]].dropna().unique(),
      )
      q2 = st.selectbox(
          "Ямар зүйл дээр суухад цаг хурдан өнгөрдөг вэ?:",
          df_survey[cols_s[3]].dropna().unique(),
      )
      q3 = st.slider("Логик ба кодод сонирхолтой байдал (1-5):", 1, 5, 3)
    with col_m2:
      q4 = st.slider("Стресс даах чадвар (1-5):", 1, 5, 3)
      q5 = st.slider("Шинэ IT/AI хэрэгсэл сурах хурд (1-5):", 1, 5, 3)
      q6 = st.selectbox(
          "Дуртай бөгөөд ахицтай хичээл:",
          df_survey[cols_s[13]].dropna().unique(),
      )

    if st.button("🤖 ML Загвараар Таамаглах", type="primary"):
      try:
        # Encode user input
        v1 = (
            encoders[cols_s[2]].transform([q1])[0]
            if q1 in encoders[cols_s[2]].classes_
            else 0
        )
        v2 = (
            encoders[cols_s[3]].transform([q2])[0]
            if q2 in encoders[cols_s[3]].classes_
            else 0
        )
        v6 = (
            encoders[cols_s[13]].transform([q6])[0]
            if q6 in encoders[cols_s[13]].classes_
            else 0
        )

        input_data = [[v1, v2, q3, q4, q5, v6]]
        prediction = ml_model.predict(input_data)[0]
        predicted_label = target_encoder.inverse_transform([prediction])[0]

        st.success(
            f"🎉 **ML Загварын Таамаглал:** Танд хамгийн тохиромжтой салбар бол"
            f" **{predicted_label}** юм!"
        )
      except Exception as err:
        st.error(f"Таамаглахад алдаа гарлаа: {err}")

  st.markdown("---")
  st.subheader("🔍 ЭЕШ-ын Датанаас Сургууль Хайх (Eysh Predictor Data)")

  if not df_eysh.empty:
    search_input = st.text_input("Сургууль эсвэл мэргэжлээр хайх:", "")
    if search_input:
      filtered_df = df_eysh[
          df_eysh["program_name"].str.contains(search_input, case=False)
          | df_eysh["university_name"].str.contains(search_input, case=False)
      ]
      st.dataframe(
          filtered_df[[
              "university_name",
              "program_name",
              "exam_subject_1",
              "threshold_score_1",
          ]],
          use_container_width=True,
      )
    else:
      st.dataframe(
          df_eysh[[
              "university_name",
              "program_name",
              "exam_subject_1",
              "threshold_score_1",
          ]].head(10),
          use_container_width=True,
      )
with tab4:
  st.header("💬 AI Сэтгэл Зүйч & Зөвлөх Чатбот")
  user_msg = st.text_input("Таны асуулт:")

  if st.button("Илгээх"):
    if user_msg:
      st.write(f"**Та:** {user_msg}")
      msg_lower = user_msg.lower()

      if "стресс" in msg_lower or "ядар" in msg_lower:
        st.write(
            "**EduPath AI:** Стрессдэж байвал түр завсарлага аваад гүнзгий"
            " амьсгаа аван 10 минут алхаарай."
        )
      elif "python" in msg_lower or "код" in msg_lower or "ml" in msg_lower:
        st.write(
            "**EduPath AI:** Python болон Machine Learning-ийг өдөр бүр бага"
            " багаар дадлага хийж сураарай[cite: 3, 4]!"
        )
      elif "эеш" in msg_lower or "сургууль" in msg_lower:
        st.write(
            "**EduPath AI:** '3.  Мэргэжил Таамаглагч' таб руу орж сургуулийн"
            " босго оноогоо хараарай[cite: 3, 4]."
        )
      else:
        st.write(
            "**EduPath AI:** Амжилт хүсье! Зорилгодоо хүрэхийн тулд өдөр бүр"
            " хичээгээрэй."
        )
