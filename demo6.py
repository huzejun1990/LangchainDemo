# @Author : huzejun
# @Time : 2025/1/22 2:43


import os
from operator import itemgetter

import bs4
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.indexes import vectorstore
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables import RunnableWithMessageHistory, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
# model = ChatOpenAI(model='gpt-4-turbo')
model = ChatOpenAI(model='gpt-3.5-turbo')

# model = ChatOpenAI(model='gpt-4o-mini')

# sqlalchemy 初始化mysql数据库的连接
HOSTNAME = '127.0.0.1'
PORT = '3306'
DATABASE = 'db2024'
USERNAME = 'root'
PASSWORD = '123456'
# mysqlclient驱动URL
MYSQL_URI = 'mysql+mysqldb://{}:{}@{}:{}/{}?charset=utf8mb4'.format(USERNAME, PASSWORD, HOSTNAME, PORT, DATABASE)

db = SQLDatabase.from_uri(MYSQL_URI)

# 测试连接是否成功
# print(db.get_usable_table_names())

# print(db.run('select * from employee limit 10;'))


# 直接使用大模型和数据库整合
test_chain = create_sql_query_chain()
# resp = test_chain.invoke({'question':'请问员工表中有多少条数据？'})
# print(resp)

answer_prompt = PromptTemplate.from_template(
    """给定以下用户问题、SQL语句和SQL执行后的结果，回答用户问题。
    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    回答: """
)

#创建一个执行sql语句的工具
execute_sql_tool = QuerySQLDataBaseTool(db=db)

# 1、初始化生成SQL语句，2、执行SQL

# 2、模板
chain = (RunnablePassthrough.assign(query=test_chain).assign(result=itemgetter('query') | execute_sql_tool)
        | answer_prompt
        | model
        | StrOutputParser
        )

rep = chain.invoke({'question':'请问员工表中有多少条数据？'})
print(rep)