
import requests
import json


def main():
    url = "https://qianfan.baidubce.com/v2/chat/completions"

    payload = json.dumps({
        "model": "ernie-3.5-8k",
        "messages": [
            {
                "role": "system",
                "content": "平台助手"
            },
            {
                "role": "user",
                "content": "你好,请介绍一下自己"
            }
        ]
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'bce-v3/ALTAK-dKYQG0wy3R0vZfmdErj7U/d95f72c1c57844953a30ffa73d47ee15955cf0ed'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)


if __name__ == '__main__':
    main()
