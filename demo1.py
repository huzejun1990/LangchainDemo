# @Author : huzejun
# @Time : 2025/1/3 18:55


import os

import fastapi as FastAPI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langserve import add_routes


os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
# os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'

# 调用大语言模型
# 创建模型

# model = ChatOpenAI(model='gpt-4-turbo') gpt-3.5-turbo
model = ChatOpenAI(model = 'gpt-4o-mini')
# model = ChatOpenAI(model='gpt-3.5-turbo')
# model = ChatOpenAI(model.openai_api_key='demo')
# model = langchain_openai.ChatOpenAI.openai_api_key='demo'
# model = langchain_openai.chat_models ='gpt-4o-mini'

# model = ChatOpenAI.openai_api_key('demo')
# model = ChatOpenAI.model_name('gpt-4o-mini')
# model = ChatOpenAI(model='gpt-4o-mini')

# 2、准备prompt
msg = [
    SystemMessage(content='请将以下的内容翻译成意大利语'),
    HumanMessage(content='你好，请问你要去哪里？')
]
# #
# result = model.invoke(msg)
# print(result)
# print("你好")

# 简单的解析响应数据
# 3、创建返回数据的解析器
# parser = StrOutParser()
parser = StrOutputParser()
# print(parser.invoke(result))

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages({
    ('system', '请将下面的内容翻译成{language}'),
    ('user', "{text}")
})

# 4、得到链
chain = prompt_template | model | parser

# 5、直接使用chain来调用
# print(chain.invoke(msg))
print(chain.invoke({'language': 'English', 'text': '我下午还有一节课，不能去打球了。'}))

# 把我们的程序部署成服务
# 创建fastAPI的应用
app = FastAPI(title='我的Langchain服务', version='V1.0', description='使用Langchain翻译任何语句的服务器')

add_routes(
    app,
    chain,
    path="/chainDemo",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)