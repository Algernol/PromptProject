import os

file_path = "C:\\Users\\乔安\\AppData\\Roaming\\nltk_data\\tokenizers\\punkt_tab.zip\\punkt_tab\\english\\collocations.tab"

if os.path.exists(file_path):
    print("文件存在！")
else:
    print("文件不存在。")
