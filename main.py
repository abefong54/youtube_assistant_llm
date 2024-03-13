import streamlit as st
import langchain_helper as lh
import langchain_text_helper as lhtxt
import textwrap
import numpy as np
from io import StringIO
import codecs
st.title("File Analysis Assistant")

with st.sidebar: 
    with st.form(key="my_form"):

        # youtube_url_one = st.sidebar.text_area(
        #     label = "What is the FIRST Youtube video URL?",
        #     max_chars = 100,
        #     key="vid1"
        # )

        # youtube_url_two = st.sidebar.text_area(
        #     label = "What is the SECOND Youtube video URL?",
        #     max_chars = 100,
        #     key="vid2"
        # )

        query = st.sidebar.text_area(
            label = "Ask me something about the video?",
            max_chars = 90,
            key="user_query"
        )

        files = st.file_uploader("Upload txt file", type=['txt','pdf'], accept_multiple_files=True, label_visibility="visible")

        submit_button = st.form_submit_button(label="Submit")

# if youtube_url_one and youtube_url_two:
#     if query:

#         # create vector
#         vector_one = lh.create_vector_db_from_youtube_url(youtube_url_one)
#         vector_two = lh.create_vector_db_from_youtube_url(youtube_url_two)
        


#         query_response = lh.get_response_from_query_about_youtube(db1=vector_one, db2=vector_two, user_query=query) # only about vid 2
#         if query_response:
#             st.subheader("QUERY RESPONSE:")
#             st.text(textwrap.fill(query, width = 80))
#             st.text(textwrap.fill(query_response))
#     else :
#         embedding1 = lh.create_similarity_embedding_db_from_text(youtube_url_one)
#         embedding2 = lh.create_similarity_embedding_db_from_text(youtube_url_two)
#         # print(embedding1)
#         similarity_score = np.dot(embedding1, embedding2)
#         similarity_response = str(similarity_score)
#         # just return similary scoure
#         if similarity_response:
#             st.subheader("SIMILARITY SCORE:")
#             st.text(textwrap.fill(similarity_response, width = 80))



if query and files is not None:
    # read files in and create embeddings
    # for uploaded_file in files:
    # To read file into string
    bytes_data = files[0].getvalue()
    decoded_string_one = codecs.decode(bytes_data, 'utf-8')
    bytes_data = files[1].getvalue()
    decoded_string_two = codecs.decode(bytes_data, 'utf-8')


    # create vector for text searching 
    text_vector_one = lhtxt.create_vector_db_from_text(decoded_string_one)
    text_vector_two = lhtxt.create_vector_db_from_text(decoded_string_two)


    # create embedding for similarity scores
    embedding1 = lh.create_similarity_embedding_db_from_text(decoded_string_one)
    embedding2 = lh.create_similarity_embedding_db_from_text(decoded_string_two)
    similarity_score = np.dot(embedding1, embedding2)
    similarity_response = str(similarity_score)
    # just return similary scoure
    if similarity_response:
        st.subheader("TEXT SIMILARITY SCORE:")
        st.text(textwrap.fill(similarity_response, width = 80))

    query_response = lhtxt.get_response_from_query_about_textfiles(file_dbs=[text_vector_one,text_vector_two], user_query=query)
    if query_response:
        st.subheader("QUERY:")
        st.text(textwrap.fill(query, width = 80))
        st.subheader("RESPONSE:")
        st.text(textwrap.fill(query_response))

