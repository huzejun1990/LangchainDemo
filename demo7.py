# @Author : huzejun
# @Time : 2025/2/21 16:39

import os
from datetime import datetime
from operator import itemgetter

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.indexes import vectorstore
from langchain_chroma import Chroma
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.prebuilt import chat_agent_executor

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
# model = ChatOpenAI(model='gpt-4-turbo')
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

persist_dir = 'chroma_data_dir'  # 存放向量数据库的目录

urls = [
    "https://www.youtube.com/watch?v=HAn9vnJy6S4",
    "https://www.youtube.com/watch?v=dA1cHGACXCo",
    "https://www.youtube.com/watch?v=ZcEMLz27sL4",
    "https://www.youtube.com/watch?v=hvAPnpSfSGo",
    "https://www.youtube.com/watch?v=EhlPDL4QrWY",
    "https://www.youtube.com/watch?v=mmBo8nlu2j0",
    "https://www.youtube.com/watch?v=rQdibOsL1ps",
    "https://www.youtube.com/watch?v=28lC4fqukoc",
    "https://www.youtube.com/watch?v=es-9MgxB-uc",
    "https://www.youtube.com/watch?v=wLRHwKuKvOE",
    "https://www.youtube.com/watch?v=ObIltMaRJvY",
    "https://www.youtube.com/watch?v=DjuXACWYkkU",
    "https://www.youtube.com/watch?v=o7C9ld6Ln-M",
]
#
# docs = [] # document的数组
# for url in urls:
#     docs.extend(YoutubeLoader.from_youtube_url(url,add_video_info=True).load())
#
# print(len(docs))
# print(docs[0])
# # 给doc添加额外的元数据： 视频发布的时间（年份）
# for doc in docs:
#     doc.metadata['publish_year'] = int(
#         datetime.datetime.strptime(doc.metadata['publish_date'],'%Y-%m-%d %H:%M:%S').strptime('%Y'))
#
# print(docs[0].metadata)
# print(docs[0].page_content[:500])   # 第一个视频的字幕内容
#
# # 根据多个doc构建向量数据库
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=30)
# split_doc = text_splitter.split_documents(docs)
# # 向量数据库的持久化
# vectorstore = Chroma.from_documents(split_doc,embeddings,persist_directory=persist_dir)   # 并且把向量数据库持久化到磁盘

#加载磁盘中的向量数据库
vectorstore = Chroma(persist_directory=persist_dir,embedding_function=)



