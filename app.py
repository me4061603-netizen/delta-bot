import re
import random
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Delta Bot | المرشد الأكاديمي",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700&display=swap');
html, body, .stApp, .stApp * {font-family:'Cairo',sans-serif;}
.stApp{direction:rtl;text-align:right;background:linear-gradient(180deg,#f2f6fc 0%,#ffffff 55%);}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDeployButton"]{display:none !important;}
.block-container{max-width:820px;padding-top:1.5rem;padding-bottom:6rem;}
.hero{background:linear-gradient(135deg,#0b2e59 0%,#1565c0 55%,#1fa3a3 100%);color:#fff;border-radius:20px;padding:26px 26px 20px;margin-bottom:18px;box-shadow:0 10px 30px rgba(11,46,89,.25);}
.hero h1{margin:0 0 8px 0;font-size:1.65rem;color:#fff;font-weight:700;line-height:1.6;padding:0;}
.hero p{margin:0;opacity:.95;font-size:1rem;color:#fff;}
.hero small{display:block;margin-top:12px;opacity:.75;font-size:.8rem;color:#fff;}
[data-testid="stChatMessage"]{border-radius:16px;padding:12px 16px;margin-bottom:10px;background:#fff;border:1px solid #e6ecf5;box-shadow:0 2px 10px rgba(11,46,89,.06);}
.stButton>button{border-radius:999px;border:1px solid #cfdcf0;background:#fff;color:#0b2e59;font-weight:600;padding:.4rem 1rem;transition:all .15s;}
.stButton>button:hover{background:#0b2e59;color:#fff;border-color:#0b2e59;}
[data-testid="stSidebar"]{direction:rtl;text-align:right;background:#0b2e59;}
[data-testid="stSidebar"] *{color:#fff !important;}
[data-testid="stSidebar"] .stButton>button{background:rgba(255,255,255,.12);border-color:rgba(255,255,255,.35);}
[data-testid="stSidebar"] .stButton>button:hover{background:#fff;}
[data-testid="stSidebar"] .stButton>button:hover *{color:#0b2e59 !important;}
[data-testid="stSidebar"] a{text-decoration:underline;}
</style>""", unsafe_allow_html=True)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
THRESHOLD = 0.50  # قلّله لو بيقول "مش لاقي" كتير، وزوّده لو بيجاوب غلط
W_EMB, W_LEX = 0.4, 0.6  # وزن الفهم بالمعنى مقابل التطابق في الحروف


def norm(t):
    """توحيد الإملاء العربي: أ/إ/آ→ا، ة→ه، ى→ي، وحذف التشكيل والرموز."""
    t = re.sub(r"[\u064B-\u0652\u0640]", "", t)
    t = re.sub("[أإآ]", "ا", t).replace("ى", "ي").replace("ة", "ه")
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ---------- الكلام العادي (سلام، ازيك، شكرا...) ----------
FILLERS = r"\b(يا باشا|يا بيه|يا غالي|يا دكتور|يا هندسه|يا صاحبي|يا معلم|يا بوت|بوت|لو سمحت|من فضلك)\b"

# (العبارات، الرد القصير، جملة الدعوة للسؤال)
SMALL_TALK_RAW = [
    (["السلام عليكم", "سلام عليكم", "السلام عليكم ورحمة الله", "السلام عليكم ورحمة الله وبركاته", "سلام", "السلام"],
     "وعليكم السلام ورحمة الله وبركاته 👋", "أهلاً بيك، تحت أمرك في أي حاجة عن المعهد."),
    (["صباح الخير", "صباح الورد", "صباحو"],
     "صباح النور ☀️", "تحت أمرك، اسألني أي حاجة عن المعهد."),
    (["مساء الخير", "مساء النور", "مساك"],
     "مساء النور 🌙", "تحت أمرك، اسألني أي حاجة عن المعهد."),
    (["اهلا", "اهلا بيك", "اهلا بيكم", "اهلا وسهلا", "هاي", "هلو", "هالو", "مرحبا", "يا هلا", "hi", "hello", "hey"],
     "أهلاً بيك! 😊", "تحب تعرف إيه عن المعهد؟"),
    (["ازيك", "عامل ايه", "اخبارك", "اخبارك ايه", "ايه الاخبار", "ايه اخبارك", "ازاي حالك", "كيف حالك", "شلونك", "ازيك عامل ايه"],
     "الحمد لله تمام 😊", "إنت عامل إيه؟ تحب تعرف إيه عن المعهد؟"),
    (["تمام", "كويس", "الحمد لله", "تمام الحمد لله", "كويس الحمد لله", "بخير", "انا تمام", "انا كويس"],
     "الحمد لله، ربنا يديك الصحة 😊", "تحب تسأل عن حاجة في المعهد؟"),
    (["شكرا", "شكرا ليك", "شكرا جزيلا", "شكرا جدا", "متشكر", "تسلم", "تسلم ايدك", "ميرسي", "مرسي", "thanks", "thank you", "جزاك الله خيرا"],
     "العفو، ولا يهمك! 😊", "لو عندك أي سؤال تاني أنا موجود."),
    (["مع السلامة", "باي", "bye", "في امان الله", "الى اللقاء", "تصبح على خير"],
     "مع السلامة! 👋", "نورتنا، ومتى احتجت أي معلومة عن المعهد أنا هنا."),
    (["انت مين", "مين انت", "انت ايه", "اسمك ايه", "انت اسمك ايه", "عرفني بنفسك", "ايه اسمك", "انت بوت", "انت روبوت"],
     "أنا Delta Bot 🎓 المرشد الأكاديمي الذكي لمعهد الدلتا العالي للهندسة والتكنولوجيا.",
     "بجاوبك على أسئلة القبول والأقسام والدراسة والخدمات الطلابية وغيرها."),
    (["انت بشر", "انت انسان", "انت بني ادم", "انت موظف", "انت حد حقيقي", "انت شخص حقيقي"],
     "أنا مرشد أكاديمي ذكي (بوت)، مش موظف بشري في المعهد 😊",
     "بس أقدر أفيدك في أي معلومة عن المعهد."),
    (["بتعمل ايه", "تعمل ايه", "تقدر تساعدني في ايه", "ايه اللي تقدر تعمله", "ساعدني", "محتاج مساعدة", "عايز مساعدة", "ممكن مساعدة", "تقدر تعمل ايه", "اسألك عن ايه"],
     "أنا تحت أمرك في أسئلة المعهد 😊",
     "تقدر تسألني عن الأقسام، القبول والأوراق المطلوبة، نظام الساعات المعتمدة والتخرج، التدريب الميداني، والخدمات الطلابية. اسأل براحتك!"),
    (["حلو", "برافو", "جميل", "ممتاز", "شاطر", "رائع", "كويس كده", "تمام كده", "عاش", "احسنت"],
     "تسلم! 😊", "ولو حابب تسأل عن حاجة تانية أنا موجود."),
]


def norm_st(t):
    t = norm(t.lower())
    t = re.sub(r"(.)\1{2,}", r"\1", t)  # ازييييك → ازيك
    t = re.sub(FILLERS, " ", t)
    return re.sub(r"\s+", " ", t).strip()


SMALL_TALK = []
for _trigs, _short, _invite in SMALL_TALK_RAW:
    for _t in _trigs:
        SMALL_TALK.append((norm_st(_t), _short, _invite))
SMALL_TALK.sort(key=lambda x: -len(x[0]))  # الأطول الأول


def small_talk(text):
    """يرجع (الرد, الباقي من السؤال, هل الرسالة كلام عادي بس؟)."""
    rest = norm_st(text)
    shorts, last_invite = [], ""
    while rest:
        hit = next(
            (x for x in SMALL_TALK if rest == x[0] or rest.startswith(x[0] + " ")),
            None,
        )
        if not hit:
            break
        if hit[1] not in shorts:
            shorts.append(hit[1])
        last_invite = hit[2]
        rest = rest[len(hit[0]):].strip()
    if not shorts:
        return "", text, False
    reply = " ".join(shorts[:2])
    if rest:  # سلام + سؤال: رد بالسلام وبعدين جاوب السؤال
        return reply, rest, False
    return f"{reply} {last_invite}", "", True



# ---------- نطاق التخصص: رد لبق على الأسئلة اللي برّه المعهد ----------
DOMAIN = [norm(w) for w in [
    "معهد", "الدلتا", "دلتا", "قسم", "اقسام", "تخصص", "قبول", "تنسيق", "تقديم", "اقدم", "اوراق",
    "مصاريف", "مصروفات", "رسوم", "دراسه", "ساعات", "ساعه", "معتمده", "تخرج", "خريج", "امتحان",
    "جدول", "نتيجه", "دكتور", "دكاتره", "عميد", "طالب", "طلاب", "شؤون", "تدريب", "شهاده", "ماده",
    "مواد", "مقرر", "ترم", "فصل", "لائحه", "كليه", "هندسه", "اتصالات", "مدني", "معماره", "عماره",
    "تواصل", "عنوان", "تليفون", "نشاط", "انشطه", "مكتبه", "معمل", "معامل", "اعتماد", "جوده",
    "بحث", "مؤتمر", "ارشاد", "حضور", "غياب", "مرشد",
]]

OFFTOPIC = [
    ([norm(w) for w in ["كوره", "كره", "لاعب", "ماتش", "مباراه", "الاهلي", "الزمالك", "ميسي", "رونالدو", "الدوري", "منتخب"]],
     ["سؤال كروي جميل 😂 بس أنا تخصصي هنا هو معهد الدلتا، أقدر أساعدك تعرف أي حاجة عن الدراسة أو الأقسام أو الخدمات الطلابية 🎓"]),
    ([norm(w) for w in ["python", "بايثون", "برمجه", "جافا", "java", "html", "كود", "c++"]],
     ["اختيار جميل والله! 😄 لكن دوري هنا إني أساعدك في كل ما يخص معهد الدلتا. تحب تعرف حاجة عن الأقسام أو المواد الدراسية؟"]),
    ([norm(w) for w in ["نكته", "نكت", "اضحكني", "هزار", "فكهني"]],
     ["😂 شكلك داخل تفصل شوية من المذاكرة! أنا موجود علشان أساعدك في استفسارات المعهد، وعندي استعداد أجاوبك على أي سؤال يخصه."]),
]
GENERIC_REDIRECT = [
    "سؤال حلو 😊 بس تخصصي هنا هو معهد الدلتا فقط. اسألني عن الأقسام أو القبول أو الدراسة أو الخدمات الطلابية وأنا تحت أمرك 🎓",
    "ده بعيد شوية عن تخصصي 😅 أنا المرشد الأكاديمي لمعهد الدلتا، وأقدر أفيدك في أي حاجة تخص المعهد. تحب تعرف عن إيه؟",
]
CLARIFY = [
    "ممكن توضحلي سؤالك أكتر؟ 😊 أقدر أجاوبك عن الأقسام، القبول، الدراسة، التدريب والخدمات الطلابية.",
]
NO_INFO = [
    "بص، المعلومة دي مش موجودة عندي حاليًا، ومش حابب أديك معلومة غير مؤكدة 😊 الأفضل تتأكد من إدارة المعهد: 0502529808 - 0502529809",
    "للأسف مش لاقي إجابة مؤكدة للسؤال ده 🙏 جرّب تسأل بصيغة تانية، ولو محتاج تأكيد رسمي ابعت للمعهد على info@dhiet.edu.eg",
    "المعلومة دي مش ضمن البيانات اللي معايا، ومش هقولك حاجة مش متأكد منها 😊 جرّب صياغة تانية للسؤال أو اسأل المعهد مباشرة.",
]
OPENERS = ["", "", "تحت أمرك 😊\n\n", "ولا يهمك، خليني أوضحلك:\n\n"]


def is_domain(query):
    n = norm(query.lower())
    return any(w in n for w in DOMAIN)


def fallback_reply(query):
    """المعلومة مش عندي (سؤال عن المعهد) أو سؤال بره التخصص."""
    if is_domain(query):
        return random.choice(NO_INFO)
    n = norm(query.lower())
    for words, pool in OFFTOPIC:
        if any(w in n for w in words):
            return random.choice(pool)
    if len(query.split()) <= 2:
        return random.choice(CLARIFY)
    return random.choice(GENERIC_REDIRECT)


def parse_knowledge(path="knowledge.txt"):
    """كل بلوك: سطور 'س:' (صيغ السؤال) ثم 'ج:' (الإجابة). البلوكات بينها سطر فاضي."""
    with open(path, encoding="utf-8") as f:
        blocks = [b for b in f.read().strip().split("\n\n") if b.strip()]
    answers, questions, q_to_ans = [], [], []
    for b in blocks:
        qs, ans_lines, in_answer = [], [], False
        for line in b.split("\n"):
            if not in_answer and line.startswith("س:"):
                qs.append(line[2:].strip())
            elif line.startswith("ج:") and not in_answer:
                in_answer = True
                ans_lines.append(line[2:].strip())
            elif in_answer:
                ans_lines.append(line)
        if qs and ans_lines:
            answers.append("\n".join(ans_lines).strip())
            for q in qs:
                questions.append(q)
                q_to_ans.append(len(answers) - 1)
    return questions, q_to_ans, answers


@st.cache_resource(show_spinner="جارٍ تجهيز المرشد الأكاديمي...")
def load_bot():
    model = SentenceTransformer(MODEL_NAME)
    questions, q_to_ans, answers = parse_knowledge()
    nq = [norm(q) for q in questions]
    emb = model.encode(nq, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.array(emb, dtype="float32"))
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(nq)
    lex = tfidf.transform(nq)
    return model, index, tfidf, lex, questions, q_to_ans, answers



WELCOME = (
    "أهلاً بيك في Delta Bot 🎓 أنا المرشد الأكاديمي الذكي لمعهد الدلتا العالي للهندسة والتكنولوجيا.\n\n"
    "اسألني عن الأقسام، القبول، الدراسة، التدريب والخدمات الطلابية، أو اختار سؤال سريع من تحت 👇"
)
QUICK = {
    "🏛️ الأقسام العلمية": "ايه الاقسام الموجودة في المعهد؟",
    "📝 طريقة التقديم": "ازاي اقدم؟",
    "📄 الأوراق المطلوبة": "الاوراق المطلوبة للتقديم ايه؟",
    "🎯 ساعات التخرج": "كام ساعة معتمدة للتخرج؟",
    "🧪 التدريب الميداني": "ازاي اتدرب في المعهد؟",
    "📞 التواصل مع المعهد": "ازاي اتواصل مع المعهد؟",
}

with st.sidebar:
    st.markdown("### 🎓 Delta Bot")
    st.markdown("**تواصل مع المعهد**")
    st.markdown(
        "📞 0502529808 - 0502529809  \n"
        "✉️ info@dhiet.edu.eg  \n"
        "🌐 [dhiet.edu.eg](https://dhiet.edu.eg/)  \n"
        "📍 المنصورة - طلخا - أول طريق المنصورة دمياط السريع"
    )
    st.divider()
    if st.button("🗑️ محادثة جديدة", use_container_width=True):
        st.session_state.msgs = [{"role": "assistant", "content": WELCOME}]
        st.session_state.last = None
        st.rerun()

st.markdown(
    """<div class="hero"><h1>🎓 Delta Bot | المرشد الأكاديمي الذكي لمعهد الدلتا العالي</h1><p>اسألني عن القبول والأقسام والدراسة والخدمات الطلابية، وهجاوبك فوراً.</p><small>المعلومات مستمدة من الموقع الرسمي للمعهد · آخر تحديث: 21 نوفمبر 2026</small></div>""",
    unsafe_allow_html=True,
)

# الواجهة بتظهر الأول، وبعدين الموديل بيتحمّل (مرة واحدة بس)
model, index, tfidf, lex, questions, q_to_ans, answers = load_bot()

if "msgs" not in st.session_state:
    st.session_state.msgs = [{"role": "assistant", "content": WELCOME}]
if "last" not in st.session_state:
    st.session_state.last = None  # (آخر سؤال، رقم آخر إجابة) لفهم الأسئلة المتتابعة


def search(q):
    nq = norm(q)
    vec = model.encode([nq], normalize_embeddings=True).astype("float32")
    emb_scores, ids = index.search(vec, len(questions))
    emb_all = np.zeros(len(questions))
    emb_all[ids[0]] = emb_scores[0]
    lex_all = (lex @ tfidf.transform([nq]).T).toarray().ravel()
    final = W_EMB * emb_all + W_LEX * lex_all
    qid = int(final.argmax())
    return float(final[qid]), qid


def get_answer(question):
    prefix, query, chat_only = small_talk(question)
    if chat_only:
        return prefix

    score, qid = search(query)
    # سؤال متابعة قصير ("وقسم الاتصالات؟"): اربطه بآخر موضوع اتكلمنا فيه
    last = st.session_state.last
    if score < THRESHOLD and last and len(query.split()) <= 5:
        s2, q2 = search(f"{last[0]} {query}")
        if s2 >= THRESHOLD and q_to_ans[q2] != last[1]:
            score, qid = s2, q2

    if score >= THRESHOLD:
        ans_id = q_to_ans[qid]
        answer = answers[ans_id]
        st.session_state.last = (query, ans_id)
        if not prefix and len(answer) > 80:
            answer = random.choice(OPENERS) + answer
    else:
        answer = fallback_reply(query)
    return f"{prefix}\n\n{answer}" if prefix else answer


question = st.chat_input("اكتب سؤالك هنا...")
if st.session_state.get("pending"):
    question = st.session_state.pop("pending")

if question:
    st.session_state.msgs.append({"role": "user", "content": question})
    with st.spinner("لحظة وهجاوبك..."):
        reply = get_answer(question)
    st.session_state.msgs.append({"role": "assistant", "content": reply})

if len(st.session_state.msgs) == 1:
    st.markdown("**أسئلة سريعة:**")
    cols = st.columns(3)
    for i, (label, q) in enumerate(QUICK.items()):
        if cols[i % 3].button(label, key=f"quick{i}", use_container_width=True):
            st.session_state.pending = q
            st.rerun()

for m in st.session_state.msgs:
    avatar = "🎓" if m["role"] == "assistant" else "🧑‍🎓"
    with st.chat_message(m["role"], avatar=avatar):
        st.markdown(m["content"])
