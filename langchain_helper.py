from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS # Library from META for efficient similarity search
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

gpt_template = """
    You are a helpful Youtube assistnat that can help answer questions about that ideos based on the videos transcript. 
    Answer the following question: {question}  
    By searching  the following video transcript: {docs}

    Only use the factual information from the transcript to answer the question.
    If you dont feel like you have enough information to answer, say "I don't know."
Your answers should be detailed.
"""


video_url = "https://www.youtube.com/watch?v=pfW2pQBwx6A"
def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    # TAKE THE LINES FROM THE TRANSCRIPT, AND WE NEED TO CHUNK IT
    # WE DO THIS SO THAT THE OPENAPI CAN READ ALL THE DATA WITHOUT GOING OVER OPENAPI SIZE LIMITS
    # WE TAKE THOSE CHUNKS AND STORE THEM AS VECTOR STORES
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    # FAISS HELPS US DO THE SIMILARITY SEARCH
    db = FAISS.from_documents(docs, embeddings)

    # return an instance of those chunks
    return db


def get_response_from_query(db, user_query, k_similarity=4):
    # llm we chose, text-davinci, can handle 4097 tokens
    # k_similarity =since each document chunk is 1000, we can search roughly 4 documents, so our k should be set to 4


    # search the query relevant documents
    # what did they say about X in the video
    # the search will search only the document that is relevant to our query
    docs = db.similarity_search(user_query, k_similarity)
    docs_page_content = " ".join([d.page_content for d in docs])


    llm = OpenAI(model="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables = ["query","docs"],
        template = gpt_template
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=user_query, docs=docs_page_content)
    response = response.replace("\n", "")

    return response


print(create_vector_db_from_youtube_url(video_url))