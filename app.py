
import os
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(
    page_title="AI 비서",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 비서")
st.write("음성 또는 텍스트로 질문하시면 답변해 드립니다.")

# 역할 지정
sys_role = """
당신은 다양한 업무를 처리하는 AI 서비스입니다.
개인 비서처럼 친절하게 응대합니다.

찾아낼 수 없는 질문은 임의로 정답을 만들지 않습니다.
필요한 정보가 부족하면 김대리에게 관련 자료를 요청합니다.

업로드된 파일이 있다면 해당 내용을 우선 참고하여 답변합니다.
여러 파일이 있다면 파일명을 구분하여 내용을 분석합니다.
이전 대화의 맥락을 유지하면서 후속 질문에 답변합니다.
"""

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_text" not in st.session_state:
    st.session_state.file_text = ""

if "file_names" not in st.session_state:
    st.session_state.file_names = []

# 파일 업로드
uploaded_files = st.file_uploader(
    "파일 업로드",
    type=["txt", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    file_text = ""
    file_names = []

    for uploaded_file in uploaded_files:
        try:
            text = uploaded_file.getvalue().decode("utf-8-sig")

        except UnicodeDecodeError:
            st.error(
                f"{uploaded_file.name}: UTF-8 형식으로 저장해주세요."
            )
            st.stop()

        file_text += f"\n\n[파일명: {uploaded_file.name}]\n{text}"
        file_names.append(uploaded_file.name)

    # 파일 목록이나 내용이 변경되면 대화 초기화
    if (
        st.session_state.file_names != file_names
        or st.session_state.file_text != file_text
    ):
        st.session_state.messages = []

    st.session_state.file_text = file_text
    st.session_state.file_names = file_names

    st.success(f"{len(uploaded_files)}개 파일이 등록되었습니다.")

else:
    # 모든 파일이 제거되면 파일 내용과 대화 초기화
    if st.session_state.file_text:
        st.session_state.file_text = ""
        st.session_state.file_names = []
        st.session_state.messages = []

# 기존 대화 표시
st.subheader("💬 대화 기록")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 음성 입력
with st.form("task_form"):
    audio_value = st.audio_input("🎤 음성으로 질문하기")
    submit = st.form_submit_button("음성 질문 실행")

# 텍스트 입력
user_input = st.chat_input("메시지를 입력하세요...")

# 입력 변수 초기화
user_question = None

# 1. 음성 입력 처리
if submit:
    if audio_value is None:
        st.warning("음성을 녹음해주세요.")
        st.stop()

    try:
        with st.spinner("음성을 인식하고 있습니다..."):
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=("audio.wav", audio_value.getvalue(), "audio/wav")
            )

        user_question = transcript.text

    except Exception as e:
        st.error(f"음성 인식 오류: {e}")
        st.stop()

# 2. 텍스트 입력 처리
elif user_input:
    user_question = user_input

# 3. 질문이 입력되었을 때만 실행
if user_question:

    # 시스템 메시지 구성
    system_content = sys_role

    if st.session_state.file_text:
        system_content += (
            "\n\n다음 파일 내용을 참고하세요:\n"
            + st.session_state.file_text
        )

    # 현재 사용자 질문 출력
    with st.chat_message("user"):
        st.write(user_question)

    # 시스템 메시지 + 이전 대화 + 현재 질문
    input_messages = [
        {"role": "system", "content": system_content}
    ]

    input_messages.extend(st.session_state.messages)

    input_messages.append({
        "role": "user",
        "content": user_question
    })

    try:
        # LLM 호출
        with st.spinner("답변을 준비하고 있습니다..."):
            response = client.responses.create(
                model="gpt-4o-mini",
                input=input_messages
            )

            answer = response.output_text

        # AI 답변 출력
        with st.chat_message("assistant"):
            st.write(answer)

        # 대화 기록 저장
        st.session_state.messages.append({
            "role": "user",
            "content": user_question
        })

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        # 최근 10개 질문·답변 쌍 유지
        st.session_state.messages = st.session_state.messages[-20:]

    except Exception as e:
        st.error(f"오류가 발생했습니다: {e}")
