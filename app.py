
import base64
import io
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ==========================================================
# 1. 기본 설정 및 이미지 경로
# ==========================================================

st.set_page_config(
    page_title="AIVLE Mate",
    page_icon="🥠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기존 DB 경로 및 컬렉션 그대로 유지
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

BASE = Path(__file__).resolve().parent

# Google Drive의 PROJECT4 폴더
DRIVE = Path("/content/drive/MyDrive/PROJECT4")


def find_image(*names):
    for folder in (BASE / "assets", DRIVE):
        for name in names:
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                path = folder / f"{name}{ext}"
                if path.is_file():
                    return path
    return None


LOGO = find_image(
    "AIVLE 로고",
    "AIVLE  로고",
    "투명 배경 AIVLE 로고 복원"
)

COOKIE_CLOSED = find_image(
    "닫힌 포춘",
    "닫힌 포춘 쿠키",
    "fortune_closed"
)

COOKIE_OPEN = find_image(
    "열린 포춘",
    "여린 포춘",
    "열린 포춘 쿠키",
    "fortune_open"
)


def image_css_url(path, max_width=460):
    """
    이미지를 CSS 배경으로 변환.
    포춘쿠키와 로고 이미지 자체를 클릭 가능한 버튼으로 사용.
    """
    if not path:
        return None

    from PIL import Image

    with Image.open(path) as im:
        im.thumbnail((max_width, max_width))

        output = io.BytesIO()

        im.save(
            output,
            format="PNG",
            optimize=True
        )

    return (
        "data:image/png;base64,"
        + base64.b64encode(output.getvalue()).decode()
    )


# ==========================================================
# 2. 기존 RAG 함수 (변경 없음)
# ==========================================================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content for doc in docs
    )


@st.cache_resource
def get_retriever():

    embedding = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embedding,
        persist_directory=CHROMA_DIR
    )

    return vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )


@st.cache_resource
def get_chain():

    retriever = get_retriever()

    s_msg = """
    1. 반드시 제공된 문서 내용(context)에 근거하여 답변하세요.
    2. 문서에 없는 내용은 추측하지 말고 "업로드된 문서에서 확인되지 않습니다."라고 답하세요.
    3. 답변은 간결하고 명확하게 작성하세요.
    [문서]
    {context}
    """

    h_msg = "{input}"

    messages = [
        ("system", s_msg),
        ("human", h_msg)
    ]

    prompt = ChatPromptTemplate.from_messages(messages)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0
    )

    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================================
# 3. 파일 업로드 기능
# ==========================================================

@st.cache_resource
def get_attachment_chain():

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            반드시 제공된 문서에 근거해 답변하세요.

            문서에 없으면
            '업로드된 문서에서 확인되지 않습니다.'
            라고 답하세요.

            [문서]
            {context}
            """
        ),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0
    )

    return prompt | llm | StrOutputParser()


def text_from_files(files):

    results = []

    for name, data in files:

        ext = Path(name).suffix.lower()

        try:

            if ext == ".txt":

                content = data.decode("utf-8-sig")

            elif ext == ".pdf":

                from pypdf import PdfReader

                pdf = PdfReader(io.BytesIO(data))

                content = "\n".join(
                    page.extract_text() or ""
                    for page in pdf.pages
                )

            elif ext == ".pptx":

                from pptx import Presentation

                ppt = Presentation(io.BytesIO(data))

                content = "\n".join(
                    shape.text
                    for slide in ppt.slides
                    for shape in slide.shapes
                    if shape.has_text_frame
                )

            else:
                continue

            results.append(
                f"[첨부파일: {name}]\n{content[:30000]}"
            )

        except Exception as exc:

            st.warning(
                f"{name} 파일 읽기 실패: {exc}"
            )

    return "\n\n".join(results)


def answer_stream(question, files):

    # 첨부파일이 없으면 기존 RAG 사용
    if not files:
        return get_chain().stream({
            "input": question
        })

    # 첨부파일이 있으면 기존 RAG 문맥과 결합
    docs = get_retriever().invoke(question)

    context = (
        format_docs(docs)
        + "\n\n"
        + text_from_files(files)
    )

    return get_attachment_chain().stream({
        "context": context,
        "input": question
    })


# ==========================================================
# 4. 포춘쿠키 LLM
# ==========================================================

def korean_today():
    return datetime.now(
        ZoneInfo("Asia/Seoul")
    )


def fortune_message():

    # 기존 RAG와 분리된 LLM 호출
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9
    )

    response = llm.invoke(
        """
        당신은 KT AIVLE School 교육생들을 위한
        포춘쿠키 응원 메시지 작가입니다.

        다음 규칙을 따르세요.

        1. 학습, 도전, 성장, 휴식 중 하나를 주제로 합니다.
        2. 재치 있고 따뜻한 문구를 작성합니다.
        3. 한국어 20자 이내로 작성합니다.
        4. 응원 문구 하나만 출력합니다.
        5. 날짜, 따옴표, 설명, 접두사는 출력하지 않습니다.
        """
    )

    return (
        response.content.strip().strip('"“”')
        or "오늘도 충분히 잘하고 있어!"
    )


# ==========================================================
# 5. 세션 상태
# ==========================================================

if "chats" not in st.session_state:
    st.session_state.chats = {}

if "chat_id" not in st.session_state:
    st.session_state.chat_id = None

if "deleting" not in st.session_state:
    st.session_state.deleting = False

if "fortune_open" not in st.session_state:
    st.session_state.fortune_open = False

if "fortune_by_date" not in st.session_state:
    st.session_state.fortune_by_date = {}

if "queued_question" not in st.session_state:
    st.session_state.queued_question = None


def new_chat():

    cid = uuid4().hex

    st.session_state.chats[cid] = {
        "title": "새 대화",
        "messages": [],
        "files": []
    }

    st.session_state.chat_id = cid
    st.session_state.deleting = False
    st.session_state.queued_question = None


if st.session_state.chat_id not in st.session_state.chats:
    new_chat()


def toggle_cookie():
    st.session_state.fortune_open = (
        not st.session_state.fortune_open
    )


# ==========================================================
# 6. 이미지 CSS 준비
# ==========================================================

cookie_url = image_css_url(
    COOKIE_OPEN
    if st.session_state.fortune_open
    else COOKIE_CLOSED
)

logo_url = image_css_url(
    LOGO,
    max_width=350
)


# ==========================================================
# 7. 전체 UI 디자인
# ==========================================================

st.markdown("""
<style>

:root {
    --mint: #13c9bc;
    --ink: #172b38;
    --muted: #7b8993;
}

/* 전체 화면 */

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(
            ellipse at 75% 4%,
            #e7faf6 0%,
            transparent 46%
        ),
        #f7fafb;

    color: var(--ink);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1040px;
    padding-top: 1.25rem;
    padding-bottom: 9rem;
}


/* 사이드바 */

section[data-testid="stSidebar"] {
    background: #edf3f4;
    border-right: 1px solid #dce8e9;
    min-width: 245px;
    width: 285px !important;
}

section[data-testid="stSidebar"]
[data-testid="stSidebarContent"] {
    padding-top: 1.9rem;
}

section[data-testid="stSidebar"] button {
    border-radius: 12px;
    border: 0;
    background: transparent;
    text-align: left;
}

section[data-testid="stSidebar"] button:hover {
    background: #dcecea;
    color: #126c68;
}


/* AIVLE 로고 영역 */

.st-key-topbar {
    padding: .4rem .6rem .7rem;
    border-bottom: 1px solid #e5eeef;
    margin-bottom: 1rem;
}

.st-key-brand_button button {
    height: 55px;
    width: 160px;

    border: 0 !important;
    box-shadow: none !important;
    background-color: transparent !important;

    font-size: 0 !important;

    background-repeat: no-repeat !important;
    background-size: contain !important;
    background-position: left center !important;
}

.st-key-brand_button button:focus {
    box-shadow: none !important;
}


/* 환영 메시지 */

.st-key-welcome {
    padding: 1.4rem 0 .6rem;
    text-align: center;
}

.st-key-welcome h1 {
    font-size: 2.3rem;
    line-height: 1.25;
    letter-spacing: -.04em;
    color: #183a43;
}

.st-key-welcome p {
    color: #728892;
    font-size: 1rem;
}


/* 포춘쿠키 카드 */

.st-key-fortune_card {
    border: 1px solid #d5eeeb;
    border-radius: 25px;
    padding: 1.1rem 1.2rem;

    background: linear-gradient(
        125deg,
        #ffffff 0%,
        #effcf9 100%
    );

    box-shadow:
        0 10px 32px rgba(24,117,112,.06);

    margin: 1rem auto 1.4rem;
}

.st-key-cookie_button button {
    height: 135px;
    width: 200px;

    border: 0 !important;
    background-color: transparent !important;

    background-size: contain !important;
    background-position: center !important;
    background-repeat: no-repeat !important;

    box-shadow: none !important;
    font-size: 0 !important;

    transition: transform .22s ease;
}

.st-key-cookie_button button:hover {
    transform: scale(1.07);
    background-color: transparent !important;
    border: 0 !important;
}

.st-key-cookie_button button:focus {
    box-shadow: none !important;
}

.st-key-fortune_card p {
    color: #55747b;
}


/* 추천 질문 카드 */

.st-key-suggestion_area button {
    background: #fff !important;
    border: 1px solid #dfe9eb !important;

    border-radius: 16px !important;
    padding: 1.15rem !important;

    min-height: 95px;

    color: #34515a !important;
    text-align: left;

    box-shadow:
        0 5px 16px rgba(22,71,72,.04);

    white-space: normal;
}

.st-key-suggestion_area button:hover {
    border-color: #62d9cd !important;
    background: #f3fffc !important;
}


/* 채팅 메시지 */

[data-testid="stChatMessage"] {
    background: rgba(255,255,255,.72);

    border: 1px solid #e8eff1;
    border-radius: 17px;

    margin: .6rem 0;
    padding: .65rem .8rem;
}


/* 채팅 입력창 */

[data-testid="stChatInput"] {
    border-radius: 23px;

    box-shadow:
        0 8px 30px rgba(23,82,79,.12);

    border: 1px solid #d9e8e7;
    background: #fff;
}

[data-testid="stChatInputSubmitButton"] button {
    background: #70eee0 !important;
    border-radius: 50% !important;
    color: #124d4a !important;
}


/* 업로드 영역 */

.st-key-upload_box {
    margin-top: 1.2rem;
    padding: 0 .4rem;
}

.st-key-upload_box [data-testid="stExpander"] {
    background: rgba(255,255,255,.8);
    border: 1px solid #e5eeee;
    border-radius: 13px;
}


/* 모바일 */

@media(max-width:700px) {

    .st-key-welcome h1 {
        font-size: 1.8rem;
    }

    .st-key-suggestion_area button {
        min-height: 70px;
    }

    section[data-testid="stSidebar"] {
        width: 260px !important;
    }
}

</style>
""", unsafe_allow_html=True)


# 이미지 버튼 CSS
if logo_url:

    st.markdown(
        f"""
        <style>
        .st-key-brand_button button {{
            background-image: url('{logo_url}') !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


if cookie_url:

    st.markdown(
        f"""
        <style>
        .st-key-cookie_button button {{
            background-image: url('{cookie_url}') !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ==========================================================
# 8. 사이드바
# ==========================================================

with st.sidebar:

    st.markdown("### 대화 기록")

    st.caption(
        "AIVLE Mate · 나만의 학습 파트너"
    )

    if st.button(
        "＋ 새 채팅",
        use_container_width=True,
        type="primary"
    ):

        new_chat()
        st.rerun()

    st.divider()

    for cid, conversation in reversed(
        list(st.session_state.chats.items())
    ):

        if st.session_state.deleting:

            st.checkbox(
                conversation["title"],
                key=f"delete_{cid}"
            )

        else:

            marker = (
                "●  "
                if cid == st.session_state.chat_id
                else "▸  "
            )

            if st.button(
                marker + conversation["title"],
                key=f"chat_{cid}",
                use_container_width=True
            ):

                st.session_state.chat_id = cid
                st.session_state.queued_question = None
                st.rerun()

    st.divider()

    if st.session_state.deleting:

        selected = [
            cid
            for cid in st.session_state.chats
            if st.session_state.get(
                f"delete_{cid}", False
            )
        ]

        if st.button(
            "🗑 선택 삭제" if selected else "취소",
            use_container_width=True
        ):

            for cid in selected:
                st.session_state.chats.pop(
                    cid, None
                )

            st.session_state.deleting = False

            if (
                st.session_state.chat_id
                not in st.session_state.chats
            ):
                new_chat()

            st.rerun()

    else:

        if st.button(
            "🗑 대화 삭제",
            use_container_width=True
        ):

            st.session_state.deleting = True
            st.rerun()


# ==========================================================
# 9. AIVLE 로고
# ==========================================================

with st.container(key="topbar"):

    with st.container(key="brand_button"):

        st.button(
            "AIVLE · 새 채팅",
            on_click=new_chat,
            key="logo_button",
            help="새 채팅 시작"
        )


# ==========================================================
# 10. 메인 채팅 및 환영 메시지
# ==========================================================

chat = st.session_state.chats[
    st.session_state.chat_id
]

if not chat["messages"]:

    with st.container(key="welcome"):

        st.markdown(
            "# 오늘도, 한 걸음 더 성장해요 ✨"
        )

        st.markdown(
            "AIVLE Mate와 함께 궁금한 것을 "
            "편하게 물어보세요."
        )


# ==========================================================
# 11. 포춘쿠키 카드
# ==========================================================

with st.container(key="fortune_card"):

    col_cookie, col_copy = st.columns(
        [1, 3],
        vertical_alignment="center"
    )

    with col_cookie:

        with st.container(key="cookie_button"):

            st.button(
                "🥠 포춘쿠키 열기/닫기",
                on_click=toggle_cookie,
                key="cookie_toggle",
                help="포춘쿠키 열기 또는 닫기"
            )

    with col_copy:

        st.markdown(
            "**🥠 오늘의 포춘쿠키**"
        )

        if st.session_state.fortune_open:

            today = korean_today()

            day_key = today.strftime(
                "%Y-%m-%d"
            )

            # 같은 세션에서 하루 한 번 생성
            if (
                day_key
                not in st.session_state.fortune_by_date
            ):

                with st.spinner(
                    "오늘의 응원 찾는 중..."
                ):

                    try:

                        message = fortune_message()

                    except Exception:

                        message = (
                            "오늘도 한 걸음 성장 중!"
                        )

                    st.session_state.fortune_by_date[
                        day_key
                    ] = message

            st.markdown(
                f"**{today:%m/%d} 오늘의 응원: "
                f"{st.session_state.fortune_by_date[day_key]}**"
            )

            st.caption(
                "쿠키를 다시 누르면 닫혀요"
            )

        else:

            st.markdown(
                "작은 응원이 필요할 땐 "
                "쿠키를 눌러보세요."
            )

            st.caption(
                "하루 한 문장 · 오늘의 행운"
            )


# ==========================================================
# 12. 기존 대화 출력
# ==========================================================

for msg in chat["messages"]:

    role = (
        "user"
        if isinstance(msg, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):

        st.markdown(
            msg.content
        )


# ==========================================================
# 13. 추천 질문 카드
# ==========================================================

if not chat["messages"]:

    st.markdown(
        "##### 이런 질문은 어때요?"
    )

    with st.container(
        key="suggestion_area"
    ):

        left, right = st.columns(2)

        with left:

            if st.button(
                "📘 선배들의 학습 꿀팁이 궁금해.",
                use_container_width=True
            ):

                st.session_state.queued_question = (
                    "선배들의 학습 꿀팁이 궁금해."
                )

                st.rerun()

        with right:

            if st.button(
                "☕ 점심 먹고 너무 졸려. "
                "깨는 방법 추천해줘.",
                use_container_width=True
            ):

                st.session_state.queued_question = (
                    "점심 먹고 너무 졸려 "
                    "잠에서 깨는 방법 추천해줘."
                )

                st.rerun()


# ==========================================================
# 14. 파일 업로드
# ==========================================================

with st.container(
    key="upload_box"
):

    with st.expander(
        "＋ 학습 자료 첨부하기 · PDF / TXT / PPTX"
    ):

        uploaded = st.file_uploader(
            "첨부할 파일",
            type=["pdf", "txt", "pptx"],
            accept_multiple_files=True,
            key=f"upload_{st.session_state.chat_id}"
        )

        chat["files"] = [
            (file.name, file.getvalue())
            for file in (uploaded or [])
        ]

        if chat["files"]:

            st.caption(
                "첨부됨: "
                + ", ".join(
                    name
                    for name, _ in chat["files"]
                )
            )


# ==========================================================
# 15. 채팅 입력 및 스트리밍 응답
# ==========================================================

question = st.chat_input(
    "무엇이든 물어보세요. "
    "AIVLE Mate가 함께할게요 ✨"
)

# 추천 질문 버튼에서 전달된 질문
if (
    not question
    and st.session_state.queued_question
):

    question = (
        st.session_state.queued_question
    )

    st.session_state.queued_question = None


if question and question.strip():

    question = question.strip()

    # 첫 질문으로 대화 제목 자동 생성
    if chat["title"] == "새 대화":

        chat["title"] = (
            question[:18]
            + ("…" if len(question) > 18 else "")
        )

    # 사용자 질문 출력
    with st.chat_message("user"):

        st.markdown(question)

    # AI 답변 스트리밍
    with st.chat_message("assistant"):

        try:

            response = st.write_stream(
                answer_stream(
                    question,
                    chat["files"]
                )
            )

        except Exception as exc:

            response = (
                "응답을 생성하지 못했어요. "
                "API 키와 ChromaDB 연결을 "
                "확인해 주세요."
            )

            st.error(str(exc))

    # 대화 기록 저장
    chat["messages"].extend([
        HumanMessage(content=question),
        AIMessage(content=response)
    ])

    st.rerun()
