from ollama import chat

messages = [
  {
    'role': 'user',
    'content': '你的输入极限16k tokens包含多少数字多少汉字多少英文',
  },
]

for part in chat('deepseek-r1:7b', messages=messages, stream=True):
  print(part['message']['content'], end='', flush=True)

print()