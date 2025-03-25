# @Author : huzejun
# @Time : 2025/3/21 20:56
import os

from langchain.chains.combine_documents.map_reduce import MapReduceDocumentsChain
from langchain.chains.combine_documents.reduce import ReduceDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
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

# 第二种：Map-Reduce
# 第一步：切割阶段
# 每一个小docs为1000个token
text_splitter=CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000,chunk_overlap=0)
split_docs=text_splitter.split_documents(docs)

# 第二步： map阶段
map_template = """以下是一组文档(documents)
"{docs}"
根据这个文档列表，请给出总结摘要:"""
map_prompt = PromptTemplate.from_template(map_template)
map_llm_chain = LLMChain(llm=model, prompt=map_prompt)

# 第三步：reduce阶段：（combine和 最终的reduce）
reduce_template = """以下是一组总结摘要：
{docs}
将这些内容提炼成一个最终的、统一的总结摘要："""
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_llm_chain = LLMChain(llm=model, prompt=reduce_prompt)

'''
reduce的思路:
如果map之后文档的累积token数超过了 4000个，那么我们将递归地将文档以<= 4000 个token的批次传递给我们的 StuffDocumentsChain 来创建批量摘要。
一旦这些批量摘要的累积大小小于 4000 个token，我们将它们全部传递给 StuffDocumentsChain 最后一次，以创建最终摘要。
'''

# 定义一个combine的chain
combine_chain = StuffDocumentsChain(llm_chain=reduce_llm_chain, document_variable_name='docs')

reduce_chain = ReduceDocumentsChain(
    # 这是最终调用的链。
    combine_documents_chain=combine_chain,
    # 中间的汇总的链
    collapse_documents_chain=combine_chain,
    # 将文档分组的最大令牌数。
    token_max=4000

)

# 第四步：合并所有的链
map_reduce_chain = MapReduceDocumentsChain(
    llm_chain=map_llm_chain,
    reduce_documents_chain=reduce_chain,
    document_variable_name='docs',
    return_intermediate_steps=False
)

# 第五步： 调用最终的链
result = map_reduce_chain.invoke(split_docs)
print(result['output_text'])