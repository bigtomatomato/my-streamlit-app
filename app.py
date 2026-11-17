
import base64
import io
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from uuid import uuid4
from operator import itemgetter

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage


# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="AIVLE 학습 도우미",
    page_icon="🥠",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# 기존 DB 설정 유지
CHROMA_DIR = "chroma_db"
COLLECTION = "aivle_docs"

# 이미지 저장 경로
ASSETS = BASE_DIR / "assets"
DRIVE_ASSETS = Path("/content/drive/MyDrive/project04")


def find_image(*names):
    """프로젝트 폴더 또는 Google Drive에서 이미지 탐색"""
    for folder in (ASSETS, DRIVE_ASSETS):
        if not folder.is_dir():
            continue

        for name in names:
            for extension in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = folder / (name + extension)

                if candidate.is_file():
                    return str(candidate)

    return None


LOGO_IMAGE = find_image(
    "투명 배경 AIVLE 로고 복원", "AIVLE  로고", "aivle_logo"
)

FORTUNE_CLOSED_IMAGE = find_image(
    "닫힌 포춘", "닫힌 포춘 쿠키", "fortune_closed"
)

FORTUNE_OPEN_IMAGE = find_image(
    "열린 포춘", "열린 포춘 쿠키", "fortune_open"
)


# ============================================================
# 2. 포춘쿠키 LLM
# ============================================================

def today_korea():
    """서버 시간대와 관계없이 한국 시간 사용"""
    return datetime.now(ZoneInfo("Asia/Seoul"))


def generate_fortune():
    """RAG를 사용하지 않고 LLM으로 응원 문구 생성"""

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.9
    )

    response = llm.invoke(
        """
        당신은 KT AIVLE School 교육생에게
        포춘쿠키 응원 메시지를 주는 작가입니다.

        학습, 성장, 도전, 휴식 중 하나를 주제로
        자연스럽고 재치 있게 응원해주세요.

        규칙:
        1. 한국어 20자 이내의 짧은 응원 문구 하나만 작성하세요.
        2. 설명 없이 응원 문구만 출력하세요.
        3. 날짜나 '오늘의 응원:' 접두어를 출력하지 마세요.
        4. 매번 다양한 표현을 사용하세요.
        """
    )

    return (
        response.content.strip().strip('"“”')
        or "오늘도 한 걸음 성장 중!"
    )


# ============================================================
# 3. 기존 RAG 함수
# ============================================================

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


# ============================================================
# 4. 업로드 파일 처리
# ============================================================

@st.cache_resource
def get_upload_chain():

    s_msg = """
    제공된 문서 내용만을 근거로 간결하고 명확하게 답하세요.

    문서에 없는 내용은 추측하지 말고
    '업로드된 문서에서 확인되지 않습니다.'라고 답하세요.

    [문서]
    {context}
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", s_msg),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0
    )

    return prompt | llm | StrOutputParser()


def uploaded_text(files):

    collected = []

    for file in files or []:

        ext = Path(file.name).suffix.lower()
        data = file.getvalue()

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

            collected.append(
                f"[첨부파일: {file.name}]\n{content[:45000]}"
            )

        except Exception as e:
            st.warning(
                f"{file.name} 파일을 읽을 수 없습니다: {e}"
            )

    return "\n\n".join(collected)


def stream_answer(question, attachments):

    if attachments:

        docs = get_retriever().invoke(question)

        context = (
            format_docs(docs)
            + "\n\n"
            + uploaded_text(attachments)
        )

        return get_upload_chain().stream({
            "context": context,
            "input": question
        })

    return get_chain().stream({
        "input": question
    })


# ============================================================
# 5. CSS 인터페이스
# ============================================================

st.markdown("""
<style>

:root {
    --aivle: #03b9c1;
    --mint: #83f7e8;
    --side: #dedede;
}

[data-testid="stAppViewContainer"] {
    background: #ffffff;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 0.4rem;
    padding-bottom: 18rem;
    max-width: none;
}

/* 사이드바 */

section[data-testid="stSidebar"] {
    width: 28vw !important;
    min-width: 240px !important;
    max-width: 350px !important;
    background: var(--side);
    border-right: 1px solid #888;
}

section[data-testid="stSidebar"]
[data-testid="stSidebarContent"] {
    background: var(--side);
}

section[data-testid="stSidebar"] button {
    border-radius: 7px;
}

/* 채팅 상단 */

.st-key-chat_header {
    background: var(--side);
    border-bottom: 1px solid #777;
    min-height: 65px;
    padding: 8px 12px;
    position: relative;
}

.st-key-chat_header [data-testid="stImage"] {
    width: 125px;
}

.st-key-chat_header [data-testid="stButton"] {
    position: absolute;
    inset: 0;
    z-index: 2;
}

.st-key-chat_header [data-testid="stButton"] button {
    width: 150px;
    height: 53px;
    opacity: 0;
    cursor: pointer;
}

/* 포춘쿠키 */

.st-key-fortune_zone {
    position: relative;
    min-height: 55px;
}

.st-key-fortune_zone [data-testid="stButton"] {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 2;
}

.st-key-fortune_zone [data-testid="stButton"] button {
    width: 65px;
    height: 55px;
    opacity: 0;
    cursor: pointer;
}

.st-key-fortune_zone [data-testid="stImage"] {
    width: 58px;
}

/* 하단 채팅 입력창 */

.st-key-bottom_composer {
    position: fixed;
    z-index: 10;
    bottom: 0;
    left: min(28vw, 350px);
    right: 0;
    background: #fff;
    padding: 10px 24px 17px;
    border-top: 1px solid #f3f3f3;
}

.st-key-bottom_composer [data-testid="stForm"] {
    background: #ddd;
    border: 1px solid #555;
    border-radius: 40px;
    padding: 1px 7px 1px 13px;
}

.st-key-bottom_composer
[data-testid="stForm"]
[data-testid="stTextInput"] input {
    background: transparent;
    border: none;
    box-shadow: none;
    font-size: 16px;
}

.st-key-bottom_composer
[data-testid="stForm"]
[data-testid="stFormSubmitButton"] button {
    background: var(--mint);
    border: none;
    border-radius: 50%;
    min-width: 38px;
    width: 38px;
    height: 38px;
    color: #126a69;
    padding: 0;
}

.st-key-bottom_composer
[data-testid="stForm"]
[data-testid="stFormSubmitButton"] button:hover {
    background: #40e6d4;
}

/* 파일 업로드 */

.st-key-bottom_composer [data-testid="stFileUploader"] {
    background: #f4f4f4;
    padding: 7px;
    border-radius: 5px;
}

.st-key-bottom_composer
[data-testid="stFileUploaderDropzone"] {
    padding: 5px;
    min-height: 58px;
}

.st-key-bottom_composer
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: none;
}

.st-key-bottom_composer
[data-testid="stFileUploaderDropzone"] button {
    font-size: 12px;
}

.st-key-bottom_composer [data-testid="stButton"] button {
    text-align: left;
}

/* 모바일 */

@media(max-width:800px) {

    section[data-testid="stSidebar"] {
        width: 260px !important;
    }

    .st-key-bottom_composer {
        left: 0;
        padding: 8px 12px;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 6. 세션 상태 관리
# ============================================================

def make_chat():

    cid = uuid4().hex

    st.session_state.chats[cid] = {
        "title": "새 대화",
        "messages": []
    }

    st.session_state.current_chat = cid
    st.session_state.fortune_open = False
    st.session_state.upload_open = False

    return cid


if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None

if "delete_mode" not in st.session_state:
    st.session_state.delete_mode = False

if "selected_chats" not in st.session_state:
    st.session_state.selected_chats = set()

if "fortune_open" not in st.session_state:
    st.session_state.fortune_open = False

if "fortune_date" not in st.session_state:
    st.session_state.fortune_date = None

if "fortune_message" not in st.session_state:
    st.session_state.fortune_message = None

if "upload_open" not in st.session_state:
    st.session_state.upload_open = False

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if st.session_state.current_chat not in st.session_state.chats:
    make_chat()


# ============================================================
# 7. 채팅 관리 함수
# ============================================================

def new_chat():

    make_chat()

    st.session_state.delete_mode = False
    st.session_state.selected_chats = set()


def choose_chat(cid):

    st.session_state.current_chat = cid
    st.session_state.fortune_open = False
    st.session_state.upload_open = False


def delete_or_toggle():

    if not st.session_state.delete_mode:

        st.session_state.delete_mode = True
        st.session_state.selected_chats = set()

        return

    selected = st.session_state.selected_chats

    if selected:

        for cid in selected:
            st.session_state.chats.pop(cid, None)

        if st.session_state.current_chat not in st.session_state.chats:
            make_chat()

    st.session_state.delete_mode = False
    st.session_state.selected_chats = set()


def update_selected(cid, checked):

    chosen = set(st.session_state.selected_chats)

    if checked:
        chosen.add(cid)

    else:
        chosen.discard(cid)

    st.session_state.selected_chats = chosen


# ============================================================
# 8. 사이드바 UI
# ============================================================

with st.sidebar:

    st.markdown("**대화 이력**")

    for cid, chat in reversed(
        list(st.session_state.chats.items())
    ):

        if st.session_state.delete_mode:

            st.checkbox(
                chat["title"],
                value=(
                    cid in st.session_state.selected_chats
                ),
                key=f"select_{cid}",
                on_change=lambda chat_id=cid: update_selected(
                    chat_id,
                    st.session_state[f"select_{chat_id}"]
                )
            )

        else:

            label = (
                ("● " if cid == st.session_state.current_chat else "")
                + chat["title"]
            )

            if st.button(
                label,
                key=f"open_{cid}",
                use_container_width=True
            ):

                choose_chat(cid)
                st.rerun()

    left, right = st.columns(2)

    with left:

        st.button(
            "🗑️",
            key="delete_button",
            help="대화 이력 선택 삭제",
            on_click=delete_or_toggle
        )

    with right:

        st.button(
            "＋",
            key="new_chat_button",
            help="새 채팅",
            on_click=new_chat
        )

    if st.session_state.delete_mode:

        st.caption(
            "삭제할 대화를 체크한 뒤 🗑️를 다시 누르세요. "
            "선택이 없으면 취소됩니다."
        )


# ============================================================
# 9. 상단 AIVLE 로고
# ============================================================

with st.container(key="chat_header"):

    if LOGO_IMAGE:

        st.image(
            LOGO_IMAGE,
            width=125
        )

    else:

        st.markdown("### AIVLE")

    st.button(
        "새 채팅",
        key="logo_new_chat",
        on_click=new_chat,
        help="새 대화 시작"
    )


# ============================================================
# 10. 포춘쿠키 UI
# ============================================================

with st.container(key="fortune_zone"):

    fortune_image = (
        FORTUNE_OPEN_IMAGE
        if st.session_state.fortune_open
        else FORTUNE_CLOSED_IMAGE
    )

    if fortune_image:

        st.image(
            fortune_image,
            width=58
        )

    else:

        st.markdown("🥠")

    if st.button(
        "포춘쿠키 열기/닫기",
        key="fortune_button"
    ):

        st.session_state.fortune_open = (
            not st.session_state.fortune_open
        )

        st.rerun()


if st.session_state.fortune_open:

    today = today_korea()
    date_key = today.strftime("%Y-%m-%d")

    # 하루에 한 번만 LLM 호출 (현재 세션 기준)
    if (
        st.session_state.fortune_date != date_key
        or not st.session_state.fortune_message
    ):

        with st.spinner("오늘의 응원을 준비하는 중..."):

            try:

                st.session_state.fortune_message = (
                    generate_fortune()
                )

            except Exception:

                st.session_state.fortune_message = (
                    "오늘도 한 걸음 성장 중!"
                )

        st.session_state.fortune_date = date_key

    st.markdown(
        f"**{today.strftime('%m/%d')} 오늘의 응원: "
        f"{st.session_state.fortune_message}**"
    )


# ============================================================
# 11. 기존 대화 출력
# ============================================================

current_id = st.session_state.current_chat
chat = st.session_state.chats[current_id]

for message in chat["messages"]:

    role = (
        "user"
        if isinstance(message, HumanMessage)
        else "assistant"
    )

    with st.chat_message(role):
        st.markdown(message.content)


# ============================================================
# 12. 사용자 질문 처리 및 스트리밍
# ============================================================

if st.session_state.pending_question:

    question = st.session_state.pending_question
    st.session_state.pending_question = None

    if chat["title"] == "새 대화":

        chat["title"] = (
            question[:17]
            + ("…" if len(question) > 17 else "")
        )

    with st.chat_message("user"):
        st.markdown(question)

    attachments = chat.get("attachments", [])

    with st.chat_message("assistant"):

        try:

            response = st.write_stream(
                stream_answer(question, attachments)
            )

        except Exception as e:

            response = (
                "응답을 생성하지 못했습니다. "
                "OpenAI API 키, ChromaDB 파일 및 "
                "의존성을 확인해주세요."
            )

            st.error(f"오류: {e}")

    chat["messages"].append(
        HumanMessage(content=question)
    )

    chat["messages"].append(
        AIMessage(content=response)
    )

    st.rerun()


# ============================================================
# 13. 하단 채팅 입력창
# ============================================================

with st.container(key="bottom_composer"):

    with st.form(
        "chat_form",
        clear_on_submit=True,
        border=False
    ):

        input_col, send_col = st.columns(
            [11, 1],
            vertical_alignment="center"
        )

        with input_col:

            message_input = st.text_input(
                "메시지",
                placeholder="메시지를 입력하세요.",
                label_visibility="collapsed",
                key=f"input_{current_id}"
            )

        with send_col:

            submitted = st.form_submit_button(
                "➤",
                help="전송",
                use_container_width=True
            )

    if submitted and message_input.strip():

        st.session_state.pending_question = (
            message_input.strip()
        )

        st.rerun()


    # ========================================================
    # 14. 파일 업로드 토글
    # ========================================================

    upload_label = (
        "🔼 파일 업로드"
        if st.session_state.upload_open
        else "🔽 파일 업로드"
    )

    if st.button(
        upload_label,
        key="toggle_upload"
    ):

        st.session_state.upload_open = (
            not st.session_state.upload_open
        )

        st.rerun()

    if st.session_state.upload_open:

        attachments = st.file_uploader(
            "PDF, TXT, PPTX 파일 추가",
            type=["pdf", "txt", "pptx"],
            accept_multiple_files=True,
            key=f"file_{current_id}"
        )

        chat["attachments"] = attachments or []

        if attachments:

            st.caption(
                "첨부됨: "
                + ", ".join(
                    f.name for f in attachments
                )
                + " (현재 대화에서만 사용)"
            )


    # ========================================================
    # 15. 추천 채팅 프롬프트
    # ========================================================

    recommended_prompts = [
        "선배들의 학습 꿀팁이 궁금해.",
        "점심 먹고 너무 졸려 잠에서 깨는 방법 추천해줘."
    ]

    for prompt in recommended_prompts:

        if st.button(
            "› " + prompt,
            key=f"suggest_{prompt}",
            type="tertiary",
            use_container_width=True
        ):

            st.session_state.pending_question = prompt
            st.rerun()
