
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
