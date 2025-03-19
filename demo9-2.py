# @Author : huzejun
# @Time : 2025/3/12 21:20
import os

from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_PREFIX, SYNTHETIC_FEW_SHOT_SUFFIX
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel

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

# 生成一些结构化的数据： 5个步骤
# 1、定义数据模型
class MedicalBilling(BaseModel):
    patient_id: int  # 患者ID，整数类型
    patient_name: str  # 患者姓名，字符串类型
    diagnosis_code: str  # 诊断代码，字符串类型
    procedure_code: str  # 程序代码，字符串类型
    total_charge: float  # 总费用，浮点数类型
    insurance_claim_amount: float  # 保险索赔金额，浮点数类型


# 2、提供一些样例数据，给AI
examples = [
    {
        "example": "Patient ID: 123456, Patient Name: 张娜, Diagnosis Code: J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"
    },
    {
        "example": "Patient ID: 789012, Patient Name: 王兴鹏, Diagnosis Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"
    },
    {
        "example": "Patient ID: 345678, Patient Name: 刘晓辉, Diagnosis Code: E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"
    },
]

# 3、创建一个提示模板，用来指导AI生成符合规定的数据
openai_template = PromptTemplate(input_variables=['example'], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    examples=examples,
    example_prompt=openai_template,
    input_variables=['subject','extra']

)


# 4、创建一个结构化数据的生成器
generator = create_openai_data_generator(
    output_schema=MedicalBilling, #指定输出数据的格式
    llm=model,
    prompt=prompt_template
)

# 5、调用生成器
result = generator.generate(
    subject='医疗账单', # 指定生成数据的主题
    # extra ='名字可以是随机的，最好使用比较生僻的人名。', # 额外的指导信息
    extra ='医疗总费用呈现正态分布，最小的总费用为1000', # 额外的指导信息
    runs=10  #指定生成数据的数量
)

print(result)



