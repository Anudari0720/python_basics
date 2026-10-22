import datetime
import pandas as pd
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
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 2. DATA LOADING (Cached)
# ==========================================
@st.cache_data
def load_data():
  try:
    df_eysh = pd.read_csv("Eysh_predictor_data - Sheet1.csv")
  except Exception:
    df_eysh = pd.DataFrame()

  try:
    df_survey = pd.read_csv("data.csv - Form responses 1.csv")
  except Exception:
    df_survey = pd.DataFrame()

  return df_eysh, df_survey


df_eysh, df_survey = load_data()


# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
if "stress_result" not in st.session_state:
  st.session_state.stress_result = None
if "goal_data" not in st.session_state:
  st.session_state.goal_data = {
      "goal_name": "Python & ML Капстойн төсөл дуусгах",
      "duration": "4 долоо хоног",
  }
if "checklist" not in st.session_state:
  st.session_state.checklist = {
      "Өдөр бүр 25 мин Python/Pandas код бичих": False,
      "ЭЕШ болон Англи хэлний бэлтгэл хийх": False,
      "7-8 цаг тогтмол унтаж амарөх": False,
      "15 минут биеийн тамир эсвэл сунгалт хийх": False,
  }
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
      {
          "role": "assistant",
          "content": (
              "Сайн байна уу? Би таны **EduPath AI** ухаалаг зөвлөх байна."
              " Судалгааны болон ЭЕШ-ын бодит дата дээр үндэслэн танд туслахад"
              " бэлэн байна. Өнөөдөр юу ярилцах вэ?"
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
    "**Ахлах ангийн сурагчдын сэтгэл зүй, ЭЕШ болон карьер төлөвлөлтийн"
    " ухаалаг платформ.**"
)
st.sidebar.markdown("---")

if not df_survey.empty and not df_eysh.empty:
  st.sidebar.success(
      f"📊 **Байршсан дата:**\n- Судалгааны оролцогч: {len(df_survey)} сурагч\n-"
      f" ЭЕШ хөтөлбөр: {len(df_eysh)} бүртгэл"
  )

st.title("🎓 EduPath AI: Ухаалаг Сурагчийн Зөвлөх Платформ")
st.markdown(
    "Та өөрийн сэтгэл зүйг хянаж, 4 долоо хоногийн зорилгоо төлөвлөн, бодит"
    " датанд суурилсан мэргэжил болон сургуулиудаа судлаарай."
)

# ==========================================
# 5. MAIN TABS STRUCTURE
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Стресс ба Ачааллын Үнэлгээ",
    "🎯 Зорилго & 4 Долоо Хоногийн Төлөвлөгч",
    "🎓 Мэргэжил & ЭЕШ Сургууль Хайгч",
    "💬 AI Сэтгэл Зүйч & Карьер Чатбот",
    "📥 Карьерын Тайлан Татах",
])

# ------------------------------------------
# TAB 1: STRESS & BURNOUT DIAGNOSTIC
# ------------------------------------------
with tab1:
  st.header("📊 Сэтгэл Зүй ба Хичээлийн Ачааллын Үнэлгээ")
  st.markdown(
      "Өдөр тутмын ачаалал, нойр болон стрессийн түвшнээ үнэлж, өөртөө тохирсон"
      " зөвлөмж аваарай."
  )

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
          " маш сайн тэнцвэртэй байна. Энэ хэмнэлээ хадгалаарай!"
      )
    elif score <= 5:
      st.session_state.stress_result = "Дунд зэрэг (Анхаарах)"
      st.warning(
          "⚠️ **Стрессийн түвшин: Дунд зэрэг.** Таны хичээлийн ачаалал ихэссэн"
          " байна. Өдөрт 15-20 минут салхинд гарч, тогтмол унтаж хэвшээрэй."
      )
    else:
      st.session_state.stress_result = "Өндөр (Анхаарал татах)"
      st.error(
          "🚨 **Стрессийн түвшин: Өндөр!** Та тархиа хэт ачааллуулсан байж"
          " болзошгүй. 1-2 хоног сайн амарч, хичээлийн хуваарьтаа өөрчлөлт"
          " оруулж, зөвлөх багш эсвэл найз нөхөдтэйгээ ярилцаарай."
      )

  if st.session_state.stress_result:
    st.markdown("---")
    st.subheader("💡 Сэтгэл Зүйн Зөвлөмж ба Амралтын Горим")
    st.markdown(
        """
        * **Pomodoro Техник:** 25 минут төвлөрч ажиллаад, 5 минут бүрэн амрах.
        * **Цэнхэр дэлгэцийн хязгаарлалт:** Унтахаас 1 цагийн өмнө утсаа хол байлгах.
        * **Өдрийн тэмдэглэл хөтлөх:** Сэтгэл зовоосон асуудлаа цаасан дээр бичиж гаргах.
        """
    )

# ------------------------------------------
# TAB 2: GOAL & ROADMAP PLANNER
# ------------------------------------------
with tab2:
  st.header("🎯 Зорилго ба 4 Долоо Хоногийн Төлөвлөгч")
  st.markdown(
      "Өөрийн хүссэн зорилгоо сонгоод, алхам алхмаар хэрэгжүүлэх 4 долоо"
      " хоногийн стратегитай танилцаарай."
  )

  goal_choice = st.selectbox(
      "Үндсэн зорилгоо сонгоно уу:",
      [
          "Python & Machine Learning Capstone төсөл дуусгах",
          "ЭЕШ болон Их сургуулийн сонголт хийх",
          "KAIST болон Гадаадын их сургуулийн бэлтгэл",
          "Өдөр тутмын бүтээмж ба цагийн менежмент сайжруулах",
      ],
  )
  st.session_state.goal_data["goal_name"] = goal_choice

  st.markdown("### 📅 4 Долоо Хоногийн Стратеги төлөвлөгөө")
  col_w1, col_w2, col_w3, col_w4 = st.columns(4)

  with col_w1:
    st.markdown(
        "**1-р долоо хоног**\n* Судалгаа хийх\n* Өгөгдөл шинжлэх\n* Төлөвлөгөө гаргах"
    )
  with col_w2:
    st.markdown(
        "**2-р долоо хоног**\n* Үндсэн хэсгийг эхлүүлэх\n* Дадлага хийх\n* Алдааг засах"
    )
  with col_w3:
    st.markdown(
        "**3-р долоо хоног**\n* Нэгдсэн байдлаар нэгтгэх\n* Туршилт хийх\n* Үр дүн гаргах"
    )
  with col_w4:
    st.markdown(
        "**4-р долоо хоног**\n* Эцсийн байдлаар шалгах\n* Тайлагнах / Презентлэх"
    )

  st.markdown("---")
  st.subheader("✅ Өнөөдрийн Хийх Дадлууд (Checklist & Progress)")

  completed_count = 0
  total_items = len(st.session_state.checklist)

  for task_text in list(st.session_state.checklist.keys()):
    is_checked = st.checkbox(
        task_text, value=st.session_state.checklist[task_text]
    )
    st.session_state.checklist[task_text] = is_checked
    if is_checked:
      completed_count += 1

  progress_val = completed_count / total_items
  st.progress(progress_val)
  st.caption(
      f"Таны өнөөдрийн гүйцэтгэл: {completed_count}/{total_items}"
      f" ({int(progress_val * 100)}%)"
  )

# ------------------------------------------
# TAB 3: DETAILED CAREER & UNIVERSITY MATCHER (USING EYSH DATA)
# ------------------------------------------
with tab3:
  st.header("🎓 Мэргэжил ба ЭЕШ Сургууль Хайгч")
  st.markdown(
      "Бодит ЭЕШ-ын дата (`Eysh_predictor_data.csv`) болон сурагчдын сонирхолд"
      " тулгуурлан их сургуулийн хөтөлбөрүүдийг шүүж харна уу."
  )

  if not df_eysh.empty:
    col_f1, col_f2 = st.columns(2)
    with col_f1:
      unis = ["Бүгд"] + sorted(df_eysh["university_name"].unique().tolist())
      selected_uni = st.selectbox("Их Сургууль сонгох:", unis)
    with col_f2:
      subjs = ["Бүгд"] + sorted(df_eysh["exam_subject_1"].unique().tolist())
      selected_subj = st.selectbox("ЭЕШ-ын гол хичээл сонгох:", subjs)

    # Filtering logic
    filtered_df = df_eysh.copy()
    if selected_uni != "Бүгд":
      filtered_df = filtered_df[filtered_df["university_name"] == selected_uni]
    if selected_subj != "Бүгд":
      filtered_df = filtered_df[filtered_df["exam_subject_1"] == selected_subj]

    st.markdown(f"### 📋 Олдсон хөтөлбөрүүд ({len(filtered_df)} үр дүн)")
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
  else:
    st.warning("Eysh_predictor_data.csv файл олдсонгүй.")

  st.markdown("---")
  st.subheader("💡 Нарийвчилсан Мэргэжлийн Чиглэлүүд")
  career_selected = st.selectbox(
      "Сонирхож буй мэргэжлийн танилцуулга үзэх:",
      [
          "Computer Science & AI Engineering",
          "Brain and Cognitive Science",
          "Human-Computer Interaction (HCI)",
          "Affective Computing",
          "Biomedical Engineering",
          "Data Science",
      ],
  )

  career_db = {
      "Computer Science & AI Engineering": {
          "desc": (
              "Компьютер судлал болон хиймэл оюун ухааны үндсэн зарчмуудыг"
              " ашиглан ухаалаг систем бүтээх чиглэл."
          ),
          "skills": (
              "Python, Machine Learning, Data Structures, Algorithms, Problem"
              " Solving"
          ),
          "subjects": "Математик, Програмчлал, Физик, Англи хэл",
      },
      "Brain and Cognitive Science": {
          "desc": (
              "Хүний тархины үйл ажиллагаа, танин мэдэхүй, сэтгэл зүйг"
              " технологитой хослуулан судлах салбар."
          ),
          "skills": "Neuroscience, Python, Data Analysis, Psychology, Statistics",
          "subjects": "Биологи, Хими, Психологи, Математик",
      },
      "Human-Computer Interaction (HCI)": {
          "desc": (
              "Хүн болон компьютер хоорондын харилцан үйлчлэлийг илүү хялбар,"
              " хүний сэтгэл зүйд нийцтэй байдлаар зохион бүтээх ухагдахуун."
          ),
          "skills": (
              "UI/UX Design, User Research, Python, Prototyping, Psychology"
          ),
          "subjects": "Дизайн, Психологи, Компьютер судлал, Англи хэл",
      },
      "Affective Computing": {
          "desc": (
              "Хүний сэтгэл хөдлөлийг таньж мэдэх, хариу үйлдэл үзүүлэх чадвартай"
              " хиймэл оюун ухааны дэвшилтэт салбар."
          ),
          "skills": (
              "Machine Learning, Computer Vision, Emotion Recognition, Python"
          ),
          "subjects": "Хиймэл оюун ухаан, Сэтгэл зүй, Математик",
      },
      "Biomedical Engineering": {
          "desc": (
              "Анагаах ухаан болон инженерийн шинжлэх ухааныг холбож эрүүл мэндийн"
              " дэвшилтэт технологи бүтээх салбар."
          ),
          "skills": "Bio-signals, Python, Circuit Design, Data Analysis",
          "subjects": "Биологи, Физик, Математик, Хими",
      },
      "Data Science": {
          "desc": (
              "Өгөгдөл шинжлэх ухааны аргаар их өгөгдлөөс үнэ цэнтэй шийдвэр"
              " гаргах мэдээлэл гаргаж ирэх."
          ),
          "skills": "Pandas, NumPy, SQL, Machine Learning, Data Visualization",
          "subjects": "Математик, Статистик, Програмчлал",
      },
  }

  info = career_db[career_selected]
  st.write(info["desc"])
  st.markdown(f"**🛠 Шаардлагатай ур чадварууд:** {info['skills']}")
  st.markdown(f"**📚 Судалбал зохих хичээлүүд:** {info['subjects']}")

# ------------------------------------------
# TAB 4: AI CHATBOT COUNSELOR
# ------------------------------------------
with tab4:
  st.header("💬 AI Сэтгэл Зүйч & Карьер Зөвлөх Чатбот")
  st.markdown(
      "Та асуух зүйлээ (хичээлийн стресс, мэргэжил сонголт, Python кодчлол,"
      " ЭЕШ гэх мэт) чөлөөтэй бичнэ үү."
  )

  for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  if user_input := st.chat_input(
      "Энд асуултаа бичнэ үү (Жишээ нь: Стрессээ хэрхэн бууруулах вэ?)"
  ):
    st.session_state.chat_history.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
      st.markdown(user_input)

    query_lower = user_input.lower()
    if (
        "стресс" in query_lower
        or "ядар" in query_lower
        or "нойр" in query_lower
        or "burnout" in query_lower
    ):
      bot_response = (
          "Стресс болон ядралт мэдрэгдэж байвал та түр завсарлаад гүнзгий"
          " амьсгаа аваарай. Өдөрт хангалттай унтаж, хичээлээ жижиг хувааж"
          " хийх нь чухал шүү. Tab 1 хэсэг рүү орж стрессийн үнэлгээгээ өгөөд"
          " үзээрэй!"
      )
    elif (
        "эеш" in query_lower
        or "босго" in query_lower
        or "оноо" in query_lower
        or "сургууль" in query_lower
    ):
      bot_response = (
          "ЭЕШ-ын бэлтгэл болон их сургуулийн босго онооны хувьд Tab 3 дээр"
          " байгаа бодит датаг ашиглан өөрийн сонирхсон сургууль болон"
          " хөтөлбөрүүдийн шаардлагыг шүүж харах боломжтой."
      )
    elif "python" in query_lower or "код" in query_lower or "ml" in query_lower:
      bot_response = (
          "Python болон Machine Learning нь ирээдүйн хамгийн эрэлттэй чадваруудын"
          " нэг юм. Pandas, NumPy санг ашиглан жижиг төслүүд хийх нь таны"
          " портфолиог маш их хүчирхэгжүүлнэ."
      )
    else:
      bot_response = (
          f"Таны асуултыг ойлголоо: '{user_input}'. EduPath AI платформ нь таны"
          " зорилгод хүрэхэд туслахад бэлэн байна. Сэтгэл зүйн зөвлөмж авах,"
          " төлөвлөгөө гаргах болон ЭЕШ болон мэргэжил сонголтоо нарийвчлаарай."
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
  st.markdown(
      "Таны бөглөсөн үнэлгээ, сонгосон зорилго болон төлөвлөгөөг нэгтгэсэн тайлан"
      " файлаа татаж аваарай."
  )

  report_text = f"""========================================
EDUPATH AI - СУРАГЧИЙН УХААЛАГ ТАЙЛАН
Үүсгэсэн огноо: {datetime.date.today()}
========================================

1. СТРЕСС БА АЧААЛЛЫН ҮНЭЛГЭЭ:
- Үр дүн: {st.session_state.stress_result if st.session_state.stress_result else "Үнэлгээ хийгдээгүй байна."}

2. ЗОРИЛГО БА ТӨЛӨВЛӨГӨӨ:
- Үндсэн зорилго: {st.session_state.goal_data['goal_name']}
- Хугацаа: {st.session_state.goal_data['duration']}

3. ӨДРИЙН ДАДЛУУДЫН БИЕЛЭЛТ (Checklist):
"""
  for task, status in st.session_state.checklist.items():
    status_str = "[X] Биелсэн" if status else "[ ] Хүлээгдэж буй"
    report_text += f"- {task}: {status_str}\n"

  report_text += """
========================================
EduPath AI Capstone Project - Generated successfully.
"""

  st.text_area("Талаас харах тайлангийн текст:", report_text, height=250)

  st.download_button(
      label="📥 Тайлан файлыг татаж авах (.txt)",
      data=report_text,
      file_name="EduPath_AI_Student_Report.txt",
      mime="text/plain",
  )

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>EduPath AI © 2026 | Built"
    " with Python & Streamlit for High School Capstone Project</p>",
    unsafe_allow_html=True,
)
