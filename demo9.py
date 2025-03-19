# @Author : huzejun
# @Time : 2025/3/12 4:17
import os

from langchain_experimental.synthetic_data import create_data_generation_chain
from langchain_openai import ChatOpenAI

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.8)
# model = ChatOpenAI(model='gpt-4-turbo')

# 创建链
# chain = create_data_generation_chain(model)
chain = create_data_generation_chain(model)

# 生成数据
# result = chain(     # 给予一些关键词，随机生成一句话
#     {
#         "fields": ['蓝色','黄色'],
#         "preferences": {"style": "让它像诗歌一样。"}
#     }
# )

result = chain(     # 给予一些关键词，随机生成一句话
    {
        "fields": {"颜色": ['蓝色', '黄色']},
        "preferences": {"style": "让它像诗歌一样。"}
    }
)
print(result)
