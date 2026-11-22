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

st.title("🎓 مساعد معهد الدلتا العالي للهندسة والتكنولوجيا")
st.caption("المعلومات من الموقع الرسمي للمعهد حتى 21 نوفمبر 2026")
debug = st.sidebar.checkbox("وضع الاختبار (إظهار الدرجة)")

if "msgs" not in st.session_state:
    st.session_state.msgs = [
        {"role": "assistant", "content": "أهلاً بيك! اسألني أي حاجة عن المعهد 👋"}
    ]

for m in st.session_state.msgs:
    st.chat_message(m["role"]).write(m["content"])

if question := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.msgs.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    prefix, query, chat_only = small_talk(question)

    if chat_only:
        answer = prefix
    else:
        nqst = norm(query)
        vec = model.encode([nqst], normalize_embeddings=True).astype("float32")
        emb_scores, ids = index.search(vec, len(questions))
        emb_all = np.zeros(len(questions))
        emb_all[ids[0]] = emb_scores[0]
        lex_all = (lex @ tfidf.transform([nqst]).T).toarray().ravel()
        final = W_EMB * emb_all + W_LEX * lex_all
        qid = int(final.argmax())
        score = float(final[qid])

        if score >= THRESHOLD:
            answer = answers[q_to_ans[qid]]
        else:
            answer = "معنديش إجابة دقيقة على السؤال ده، جرّب تسأل بصيغة تانية أو كلّم المعهد: 0502529808 - 0502529809"
        if debug:
            answer += (
                f"\n\n`final={score:.2f} (معنى={emb_all[qid]:.2f}, حروف={lex_all[qid]:.2f})"
                f" | matched: {questions[qid]}`"
            )
        if prefix:
            answer = f"{prefix}\n\n{answer}"

    st.session_state.msgs.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
