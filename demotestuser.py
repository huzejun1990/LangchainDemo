# @Author : huzejun
# @Time : 2025/1/4 15:02

import os
import openai

# 设置OpenAI API密钥
# openai.api_key = os.getenv("sk-Qxt1e040220a75c18e3a2193f6b5cb0d8fb7ff3uxxI1")  # 确保你的环境变量中设置了OPENAI_API_KEY
openai.api_key = os.getenv("demo")


# 设置gpt-4o-mini模型的demo账号
openai.Model.set_default_model("gpt-4o-mini")

# 发送消息并接收响应
response = openai.Model.generate(
    prompt="Hello, who are you?",
    max_tokens=7,  # 设置最大令牌数，根据需要调整
)

print(response['choices'][0]['message']['content'])
