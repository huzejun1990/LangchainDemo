# @Author : huzejun
# @Time : 2025/1/4 15:23

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    # model="gpt-4o-mini",
    # model="GPT_4_O_MINI",

    model="gpt-4o-mini",
    # base_url="https://api.chatfire.cn/v1/",
    # base_url="https://api.xty.app/v1",
    api_key="demo",
    # api_key="sk-DbMfA18yhwgHLHPx32F48c03E1C44416B0B81d3096F1297e",  # 在这里填入你的密钥
)
# res = llm.invoke("你是谁？请你简要做一下，自我介绍？")
res = llm.invoke("今天是几月几号？")
# 今天是几月几号？
print(res)
