# @Author : huzejun
# @Time : 2025/3/15 21:14
import os

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field

os.environ['http_proxy'] = '127.0.0.1:7890'
os.environ['https_proxy'] = '127.0.0.1:7890'

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "LangchainDemo"
os.environ["LANGCHAIN_API_KEY"] = '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'
# os.environ["TAVILY_API_KEY"] = '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'

# 聊天机器人案例
# 创建模型
model = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

# class Classification(BaseModel):
#     """
#         定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
#     """
#     # 文本的情感倾向，预期为字符串类型
#     sentiment: str = Field(descripation="文本的情感")
#
#
#     # 文本的攻击性，预期为1到10的整数
#     aggressiveness: int = Field(
#         description="描述文本的攻击性,数字越大表示越攻击性"
#     )
#
#
#     # 文本使用的语言，预期为字符串类型
#     language: str = Field(description="文本使用的语言")

class Classification(BaseModel):
    """
        定义一个Pydantic的数据模型，未来需要根据该类型，完成文本的分类
    """
    # 文本的情感倾向，预期为字符串类型
    sentiment: str = Field(..., enum=["happy", "neutral", "sad"],descripation="文本的情感")


    # 文本的攻击性，预期为1到10的整数
    aggressiveness: int = Field(...,enum=[1,2,3,4,5],
        description="描述文本的攻击性,数字越大表示越攻击性"
    )


    # 文本使用的语言，预期为字符串类型
    language: str = Field(...,enum=["spanish", "english", "french", "中文", "italian"], description="文本使用的语言")

# 创建一个用于提取信息的提示模板
tagging_prompt = ChatPromptTemplate.from_template(
    """
    从以下段落中提取所需信息。
    只提取'Classification'类中提到的属性。
    段落：
    {input}
    """
)


chain = tagging_prompt | model.with_structured_output(Classification)
input_text = "中国人民大学的王教授：师德败坏，做出的事情伤天害理！"
# input_text = "Estoy increiblemente contento de haberte conocido! Creo que seremos muy buenos amigos!"
result:Classification = chain.invoke({'input': input_text})
print(result)