import re
import random
import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="مساعد معهد الدلتا", page_icon="🎓")
st.markdown(
    "<style>.stApp{direction:rtl;text-align:right}</style>",
    unsafe_allow_html=True,
)

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
     "وعليكم السلام ورحمة الله وبركاته 👋", "اتفضل اسألني أي حاجة عن المعهد."),
    (["صباح الخير", "صباح الورد", "صباحو"],
     "صباح النور ☀️", "اتفضل اسألني أي حاجة عن المعهد."),
    (["مساء الخير", "مساء النور", "مساك"],
     "مساء النور 🌙", "اتفضل اسألني أي حاجة عن المعهد."),
    (["اهلا", "اهلا بيك", "اهلا بيكم", "اهلا وسهلا", "هاي", "هلو", "هالو", "مرحبا", "يا هلا", "hi", "hello", "hey"],
     "أهلاً بيك! 😊", "اتفضل اسألني أي حاجة عن المعهد."),
    (["ازيك", "عامل ايه", "اخبارك", "اخبارك ايه", "ايه الاخبار", "ايه اخبارك", "ازاي حالك", "كيف حالك", "شلونك", "ازيك عامل ايه"],
     "الحمد لله تمام 😊", "إنت عامل إيه؟ اسألني أي حاجة عن المعهد."),
    (["تمام", "كويس", "الحمد لله", "تمام الحمد لله", "كويس الحمد لله", "بخير", "انا تمام", "انا كويس"],
     "الحمد لله، ربنا يديك الصحة 😊", "اتفضل اسألني أي حاجة عن المعهد."),
    (["شكرا", "شكرا ليك", "شكرا جزيلا", "شكرا جدا", "متشكر", "تسلم", "تسلم ايدك", "ميرسي", "مرسي", "thanks", "thank you", "جزاك الله خيرا"],
     "العفو! 😊", "لو عندك أي سؤال تاني أنا موجود."),
    (["مع السلامة", "باي", "bye", "في امان الله", "الى اللقاء", "تصبح على خير", "سلام عليكم ورحمة الله ماشي"],
     "مع السلامة! 👋", "نورتنا، وأي وقت تحب تسأل عن المعهد أنا هنا."),
    (["انت مين", "مين انت", "انت ايه", "اسمك ايه", "انت اسمك ايه", "عرفني بنفسك", "ايه اسمك", "انت بوت", "انت روبوت"],
     "أنا مساعد معهد الدلتا العالي للهندسة والتكنولوجيا 🎓",
     "بجاوب على أسئلتك عن المعهد: القبول، الأقسام، الدراسة، الخدمات وغيرها."),
    (["بتعمل ايه", "تعمل ايه", "تقدر تساعدني في ايه", "ايه اللي تقدر تعمله", "ساعدني", "محتاج مساعدة", "عايز مساعدة", "ممكن مساعدة", "تقدر تعمل ايه", "اسألك عن ايه"],
     "أقدر أساعدك في أسئلة المعهد 😊",
     "زي: الأقسام، القبول والأوراق المطلوبة، نظام الساعات المعتمدة والتخرج، التدريب الميداني، الخدمات الطلابية، والتواصل مع المعهد. اسأل براحتك!"),
    (["حلو", "برافو", "جميل", "ممتاز", "شاطر", "رائع", "كويس كده", "تمام كده", "عاش", "احسنت"],
     "تسلم! 😊", "لو عايز تسأل عن حاجة تانية أنا موجود."),
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


@st.cache_resource
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


model, index, tfidf, lex, questions, q_to_ans, answers = load_bot()

# ---------- المرشد الأكاديمي الذكي ----------
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

SYSTEM_PROMPT = """
أنت «المرشد الأكاديمي» في معهد الدلتا العالي للهندسة والتكنولوجيا 🎓.

هويتك ومهمتك:
- اسمك دائمًا: المرشد الأكاديمي، ولا تصف نفسك بأنك «مساعد» أو «شات بوت عام».
- أنت مرشد ودود ومتخصص حصريًا في كل ما يخص معهد الدلتا العالي للهندسة والتكنولوجيا.
- تحدث باللهجة المصرية الطبيعية المهذبة، بأسلوب بشوش ومريح وخفيف الظل.
- يمكنك استخدام دعابة خفيفة وإيموجي باعتدال عندما يناسب السياق.
- اجعل الإجابات واضحة ومنظمة ومختصرة بالقدر المناسب، واستخدم النقاط والعناوين عند الحاجة.

حدود التخصص:
- أجب فقط عن الأسئلة المتعلقة بمعهد الدلتا: القبول والتقديم، الأقسام والتخصصات،
  الدراسة والامتحانات، المصروفات، الأوراق، التدريب، شؤون الطلاب، الخدمات،
  الأنشطة، الإدارات، التواصل، وأي شأن آخر خاص بالمعهد.
- يمكنك الرد على التحية والشكر والمجاملات والدردشة الاجتماعية القصيرة.
- إذا كان السؤال خارج موضوع المعهد، لا تجب عنه حتى لو كنت تعرف إجابته.
  حوّل الحديث بلطف إلى تخصصك، مثل:
  «سؤال لطيف 😄 بس أنا هنا المرشد الأكاديمي لمعهد الدلتا، ومهمتي أساعدك في كل
  ما يخص المعهد. تحب تعرف حاجة عن الأقسام أو الدراسة أو التقديم؟»

مصدر المعلومات والدقة:
- سياق قاعدة بيانات المعهد المرفق مع كل سؤال هو المصدر الوحيد للمعلومات الرسمية.
- لا تستخدم معرفتك العامة لملء أي نقص في معلومات المعهد.
- لا تخترع أسماء أو أرقامًا أو مصروفات أو مواعيد أو شروطًا أو قرارات.
- إذا كان السؤال عن المعهد لكن الإجابة غير موجودة في السياق المرفق، قل بوضوح
  وبلطف إن المعلومة غير متاحة في قاعدة البيانات الحالية، واقترح التواصل مع
  المعهد للتأكد. لا تخمّن ولا تستنتج.
- إذا كان السياق لا يجيب عن السؤال بالكامل، اذكر الجزء المدعوم فقط ووضّح ما ينقص.
- لا تدّعِ أنك موظف بشري أو مسؤول رسمي أو أنك تواصلت مع إدارة المعهد.

الحوار:
- افهم الأسئلة المتتابعة بالرجوع إلى سياق المحادثة.
- لا تطلب إعادة معلومة ذكرها المستخدم بالفعل.
- إذا كان السؤال غامضًا، اسأل سؤال توضيح واحدًا.
- لا تكرر جملة «تواصل مع المعهد» إلا عندما تكون المعلومة الرسمية المطلوبة غير متاحة فعلًا.

تذكّر: الذكاء الاصطناعي يساعدك على فهم السؤال وصياغة رد لطيف، لكنه لا يوسّع
نطاق تخصصك ولا يحل محل قاعدة بيانات المعهد.
"""

def get_gemini_key():
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        return ""

def call_gemini(user_question, institute_context):
    """توليد رد لطيف مع حصر المعلومات الرسمية في سياق قاعدة بيانات المعهد."""
    import json
    import urllib.request
    import urllib.error

    api_key = get_gemini_key()
    if not api_key:
        return None, "مفتاح Gemini API غير مضبوط في Streamlit Secrets."

    context = institute_context.strip() or (
        "لا توجد إجابة مطابقة متاحة لهذا السؤال في قاعدة بيانات المعهد."
    )

    history = []
    # آخر الرسائل السابقة فقط للحفاظ على سياق الحوار
    for msg in st.session_state.msgs[-8:]:
        role = "model" if msg["role"] == "assistant" else "user"
        history.append({"role": role, "parts": [{"text": msg["content"]}]})

    current_prompt = (
        "سياق قاعدة بيانات المعهد (استخدمه وحده لأي معلومة رسمية عن المعهد):\n"
        f"{context}\n\n"
        f"رسالة المستخدم الحالية:\n{user_question}"
    )

    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": history + [
            {"role": "user", "parts": [{"text": current_prompt}]}
        ],
        "generationConfig": {
            "temperature": 0.65,
            "topP": 0.9,
            "maxOutputTokens": 650
        }
    }

    request = urllib.request.Request(
        f"{GEMINI_URL}?key={api_key}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=35) as response:
            result = json.loads(response.read().decode("utf-8"))
        parts = (
            result.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [])
        )
        answer = "\n".join(
            part.get("text", "") for part in parts if part.get("text")
        ).strip()
        if answer:
            return answer, None
        return None, "لم يرجع Gemini ردًا نصيًا."
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="ignore")
        return None, f"Gemini API error ({error.code}): {details[:400]}"
    except Exception as error:
        return None, str(error)


st.set_page_config(page_title="المرشد الأكاديمي | معهد الدلتا", page_icon="🎓")
st.markdown(
    """
    <style>
    .stApp { direction: rtl; text-align: right; }
    [data-testid="stChatMessage"] { direction: rtl; text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 المرشد الأكاديمي | معهد الدلتا العالي للهندسة والتكنولوجيا")
st.caption("معاك خطوة بخطوة في كل ما يخص المعهد 💙")

with st.sidebar:
    st.subheader("المرشد الأكاديمي")
    if get_gemini_key():
        st.success("خدمة الذكاء الاصطناعي جاهزة")
    else:
        st.warning("أضف GEMINI_API_KEY في إعدادات Secrets لتفعيل الردود الذكية.")
    debug = st.checkbox("وضع الاختبار (إظهار درجة تطابق قاعدة البيانات)")
    if st.button("مسح المحادثة"):
        st.session_state.msgs = [
            {
                "role": "assistant",
                "content": "أهلاً بيك! 😊 أنا المرشد الأكاديمي لمعهد الدلتا. اسألني عن أي حاجة تخص المعهد، وأنا معاك.",
            }
        ]
        st.rerun()

if "msgs" not in st.session_state:
    st.session_state.msgs = [
        {
            "role": "assistant",
            "content": "أهلاً بيك! 😊 أنا المرشد الأكاديمي لمعهد الدلتا. اسألني عن أي حاجة تخص المعهد، وأنا معاك.",
        }
    ]

for message in st.session_state.msgs:
    st.chat_message(message["role"]).write(message["content"])

if question := st.chat_input("اسأل المرشد الأكاديمي عن المعهد..."):
    st.session_state.msgs.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    prefix, query, chat_only = small_talk(question)

    if chat_only:
        answer = (
            f"{prefix} أنا المرشد الأكاديمي لمعهد الدلتا، وتحت أمرك في أي استفسار يخص المعهد 🎓"
        )
        st.chat_message("assistant").write(answer)
    else:
        # البحث في قاعدة بيانات المعهد أولًا
        normalized_question = norm(query)
        vector = model.encode(
            [normalized_question], normalize_embeddings=True
        ).astype("float32")
        embedding_scores, ids = index.search(vector, len(questions))
        embedding_all = np.zeros(len(questions))
        embedding_all[ids[0]] = embedding_scores[0]
        lexical_all = (
            lex @ tfidf.transform([normalized_question]).T
        ).toarray().ravel()
        final_scores = W_EMB * embedding_all + W_LEX * lexical_all
        question_id = int(final_scores.argmax())
        score = float(final_scores[question_id])

        # لا نمرر أي إجابة رسمية للنموذج إلا عند تجاوز عتبة التطابق.
        institute_context = (
            answers[q_to_ans[question_id]] if score >= THRESHOLD else ""
        )

        if get_gemini_key():
            with st.chat_message("assistant"):
                with st.spinner("ثواني، المرشد الأكاديمي بيجهزلك الرد... 😊"):
                    answer, error = call_gemini(query, institute_context)
                    if answer is None:
                        st.error("حصلت مشكلة مؤقتة أثناء تجهيز الرد.")
                        if debug:
                            st.code(error or "Unknown error")
                        answer = (
                            "معلش، حصلت مشكلة تقنية بسيطة وأنا بجهزلك الرد 😅 "
                            "جرّب تاني بعد شوية."
                        )
                    if prefix:
                        answer = f"{prefix}\n\n{answer}"
                    st.write(answer)
        else:
            # إجابة احتياطية عند عدم ضبط مفتاح API
            if institute_context:
                answer = institute_context
            else:
                answer = (
                    "المعلومة دي مش موجودة عندي حاليًا في بيانات المعهد، "
                    "ومش حابب أقولك حاجة غير مؤكدة 😊"
                )
            if prefix:
                answer = f"{prefix}\n\n{answer}"
            st.chat_message("assistant").write(answer)

        if debug:
            st.caption(
                f"درجة تطابق قاعدة البيانات: {score:.2f} | "
                f"السؤال الأقرب: {questions[question_id]}"
            )

    st.session_state.msgs.append({"role": "assistant", "content": answer})
