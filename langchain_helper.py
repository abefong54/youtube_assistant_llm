# from langchain.document_loaders import YoutubeLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.llms import OpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
# from langchain.vectorstores import FAISS # Library from META for efficient similarity search
# from dotenv import load_dotenv
# import openai

# load_dotenv()

# embeddings = OpenAIEmbeddings()

# gpt_template = """
#     You are a helpful research assistnat that can help answer questions based on the transcript from two different. 
#     Answer the following question: {question}  
#     By searching the following video transcript from video one: {docs_one}

#     and from video two: {docs_two}

#     Only use the factual information from the transcript to answer the question.
#     If you dont feel like you have enough information to answer, say "I don't know."
# Your answers should be detailed.
# """


# # video_url = "https://www.youtube.com/watch?v=pfW2pQBwx6A"


# def get_transcript_from_video(url):
#     # TAKE THE LINES FROM THE TRANSCRIPT, AND WE NEED TO CHUNK IT
#     # WE DO THIS SO THAT THE OPENAPI CAN READ ALL THE DATA WITHOUT GOING OVER OPENAPI SIZE LIMITS
#     # WE TAKE THOSE CHUNKS AND STORE THEM AS VECTOR STORES
#     loader = YoutubeLoader.from_youtube_url(url)
#     transcript = loader.load()
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     docs = text_splitter.split_documents(transcript)
#     return docs


# # ISSUES WITH THIS
# def create_similarity_embedding_db_from_text(text):
#     # CREATE EMBEDDINGS TO TEST SIMILARITY SCORES
#     return openai.embeddings.create(input = [text], model="text-embedding-ada-002").data[0].embedding


# def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
#     # FAISS HELPS US DO THE SIMILARITY SEARCH
#     docs =get_transcript_from_video(video_url)
#     db = FAISS.from_documents(docs, embeddings)
#     return db
    

# def get_response_from_query_about_youtube(db1, db2, user_query, k_similarity=4):
#     # llm we chose, text-davinci, can handle 4097 tokens
#     # k_similarity = since each document chunk is 1000, we can search roughly 4 documents, so our k should be set to 4
#     # search the query relevant documents
#     # what did they say about X in the video
#     # the search will search only the document that is relevant to our query
#     docs_one = db1.similarity_search(user_query, k_similarity)
#     docs_page_content_one = " ".join([d.page_content for d in docs_one])


#     docs_two = db2.similarity_search(user_query, k_similarity)
#     docs_page_content_two = " ".join([d.page_content for d in docs_two])

#     llm = OpenAI(model="gpt-3.5-turbo-instruct")

#     prompt = PromptTemplate(
#         input_variables = ["query","docs"],
#         template = gpt_template
#     )

#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.run(question=user_query, docs_one=docs_page_content_one, docs_two=docs_page_content_two)
#     response = response.replace("\n", "")

#     return response




# def get_response_from_query_about_textfiles(file_db, user_query, k_similarity=4):
#     # llm we chose, text-davinci, can handle 4097 tokens
#     # k_similarity = since each document chunk is 1000, we can search roughly 4 documents, so our k should be set to 4
#     # search the query relevant documents
#     # what did they say about X in the video
#     # the search will search only the document that is relevant to our query
#     docs_one = file_db.similarity_search(user_query, k_similarity)
#     docs_page_content_one = " ".join([d.page_content for d in docs_one])

#     llm = OpenAI(model="gpt-3.5-turbo-instruct")

#     prompt = PromptTemplate(
#         input_variables = ["query","docs"],
#         template = gpt_template
#     )

#     chain = LLMChain(llm=llm, prompt=prompt)
#     response = chain.run(question=user_query, docs_one=docs_page_content_one, docs_two=docs_page_content_one)
#     response = response.replace("\n", "")

#     return response


