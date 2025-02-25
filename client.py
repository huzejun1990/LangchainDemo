# @Author : huzejun
# @Time : 2025/1/4 1:17

from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable('http://127.0.0.1:8000/chainDemo/')
    print(client.invoke({'language': 'italian', 'text': '你好！'}))