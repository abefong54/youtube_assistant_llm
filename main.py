import streamlit as st
import langchain_helper as lh
import textwrap

st.title("Youtube Assistant")

with st.sidebar: 
    with st.form(key="my_form"):
        youtube_url = st.sidebar.text_area(
            label = "What is the Youtube video URL?",
            max_chars = 50
        )

        query = st.sidebar.text_area(
            label = "Ask me something about the video?",
            max_chars = 90,
            key="user_query"
        )

        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lh.create_vector_db_from_youtube_url(youtube_url)
    response = lh.get_response_from_query(db=db, user_query=query)
    st.subheader("Answer:")
    st.text(textwrap.fill(response, width = 80))