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
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.document_loaders import WebBaseLoader
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
model = ChatOpenAI(model='gpt-4-turbo')
# model = ChatOpenAI(model='gpt-3.5-turbo')

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

# 创建工具
toolkit = SQLDatabaseToolkit(db=db, llm=model)
tools = toolkit.get_tools()

# 使用agent完成整个数据库的整合
system_prompt = """
您是一个被设计用来与SQL数据库交互的代理。
给定一个输入问题，创建一个语法正确的SQL语句并执行，然后查看查询结果并返回答案。
除非用户指定了他们想要获得的示例的具体数量，否则始终将SQL查询限制为最多10个结果。
你可以按相关列对结果进行排序，以返回MySQL数据库中最匹配的数据。
您可以使用与数据库交互的工具。在执行查询之前，你必须仔细检查。如果在执行查询时出现错误，请重写查询SQL并重试。
不要对数据库做任何DML语句(插入，更新，删除，删除等)。

首先，你应该查看数据库中的表，看看可以查询什么。
不要跳过这一步。
然后查询最相关的表的模式。
"""
system_message = SystemMessage(content=system_prompt)

# 创建代理

agent_executor = chat_agent_executor.create_tool_calling_executor(model,tools,system_message)

# resp = agent_executor.invoke({'messages': [HumanMessage(content='请问: 员工表中有多少条数据？')]})
# resp = agent_executor.invoke({'messages': [HumanMessage(content='请问: 那种性别的员工人数最多？')]})
resp = agent_executor.invoke({'messages': [HumanMessage(content='哪个部门下面的员工人数最多？')]})

result = resp['messages']
print(result)
print(len(result))
# 最后一个才是真正的答案
print(result[len(result)-1])
