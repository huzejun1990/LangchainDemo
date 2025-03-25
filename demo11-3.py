# @Author : huzejun
# @Time : 2025/3/26 0:37
import os

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 创建模型
# model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0) # gpt-4o
# model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# 加载我们的文档。我们将使用 WebBaseLoader 来加载博客文章：
loader = WebBaseLoader('https://lilianweng.github.io/posts/2023-06-23-agent/')
docs = loader.load()  # 得到整篇文章


# 第三种：Refine
'''
Refine: RefineDocumentsChain 类似于map-reduce：
文档链通过循环遍历输入文档并逐步更新其答案来构建响应。对于每个文档，它将当前文档和最新的中间答案传递给LLM链，以获得新的答案。
'''
# 第一步：切割阶段
# 每一个小docs为1000个token
text_splitter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=0)
split_docs=text_splitter.split_documents(docs)


# 指定chain_type为：refine
chain = load_summarize_chain(model,chain_type='refine')

result = chain.invoke(split_docs)

# print(result['output_text'])


