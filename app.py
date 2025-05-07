import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage,AIMessage, HumanMessage
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

#streamlitの初期化
st.set_page_config(
    page_title="英語学習サポートAI",
    page_icon="🤖",
    layout="centered",
)
st.title("英語学習サポートAI")

#セッションに履歴を保存する
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        HumanMessage(content="TOEICのリスニングが苦手で..."),
        AIMessage(content="TOEICのリスニングは難しいですよね。どの部分が特に苦手ですか？"),
        HumanMessage(content="Part3の問題が特に苦手です。"),
    ]

#LLMの初期化
llm = ChatOpenAI(
    openai_api_key=st.secrets["openai_api_key"],
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=1000,
)

#プロンプトの初期化
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="あなたは英語学習をサポートするコーチです。ユーザーの目標や進捗を参考にして、最適な学習法を提案してください。"),
    MessagesPlaceholder(variable_name="chat_history"),
    HumanMessagePromptTemplate.from_template("{input}"),
])

#チェーンの作成
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    ),
)

#ユーザーからの入力を受け取る
user_input = st.chat_input("英語学習について相談してみよう！")

if user_input:
    #ユーザーの入力を履歴に追加
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    #AIからの応答を生成
    response = chain.run(input=user_input)
    
    #AIの応答を履歴に追加
    st.session_state.chat_history.append(AIMessage(content=response))
    
    #履歴を表示
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.chat_message("user").markdown(message.content)
        elif isinstance(message, AIMessage):
            st.chat_message("assistant").markdown(message.content)