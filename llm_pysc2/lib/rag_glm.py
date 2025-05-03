# Copyright 2024, LLM-PySC2 Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Thanks https://zhuanlan.zhihu.com/p/682543197


import requests
import os


# Set Knowledge Base Embedding model
def glm_dataset_setting(api_base, api_key, dataset_id, **kwargs):
  url = f"{api_base}/knowledge/{dataset_id}"
  headers = {
    'accept': '*/*',
    'Authorization': f'{api_key}',
    'Content-Type': 'application/json'
  }
  data = {
    'embedding_id': 3,
  }
  response = requests.put(url, headers=headers, json=data)
  return response


# Create document by file
def glm_document_create_by_file(api_base, api_key, dataset_id, **kwargs):
  path = str(kwargs.get('path', 'None'))
  if not os.path.isfile(path):
    raise FileExistsError(f"{path} is not a file path")
  url = f"{api_base}/document/upload_document/{dataset_id}"
  headers = {
    'accept': '*/*',
    'Authorization': f'{api_key}'}
  form_data = {
    'knowledge_type': 5,
    'custom_separator': ["------"],
    'sentence_size': 2000,
  }
  files = {'files': (os.path.basename(path), open(path, 'rb'), 'application/octet-stream')}
  response = requests.post(url, headers=headers, data=form_data, files=files)
  files['files'][1].close()
  return response


# Retrieval
def glm_retrieval(api_base, api_key, dataset_id, **kwargs):
  url = f"{api_base}/model-api/{id}/invoke"
  headers = {
    'Authorization': f'{api_key}',
    'Content-Type': 'application/json'
  }
  data = {
    "name": "11111130",
    "desc": "11111130",
    "prompt": "从文档\n\"\"\"\n{{知识}}\n\"\"\"\n中找问题\n\"\"\"\n{{用户}}\n\"\"\"\n的答案，找到答案就仅使用文档语句回答问题，找不到答案就用自身知识回答并且告诉用户该信息不是来自文档。\n\n不要复述问题，直接开始回答。",
    "temperature": 0.01,
    "top_p": 0.1,
    "knowledge_ids": [
        1855805746403495936
    ],
    "param_desc": "flexible",
    "max_token": 1024,
    "knowledge_info": {
        "model": "glm-4-plus",
        "knowledge_ids": [
            "1855805746403495936"
        ],
        "slice_config_type": "customized",
        "recall_method": "embedding",
        "recall_index_type_list": [
            {
                "knowledge_id": "1855805746403495936",
                "index_type_id": "0"
            }
        ],
        "slice_count": 8,
        "rerank_status": 0,
        "rerank_model_name": "",
        "show_recall_result": True,
        "recall_slice_splicing_method": "{{切片内容}}"
    }
}


if __name__ == "__main__":

  from pprint import pprint
  # url = 'YOUR_DIFY_API_BASE'
  # api_key = 'YOUR_DIFY_API_KEY'
  api_base = 'https://open.bigmodel.cn/api/llm-application/open'
  api_key = 'ca25d16b0e17800b88c1adbb5f171e5d.zVaP2ponziMvVcUn'
  dataset_id = '1859169518975418368'
  # document_id = 'd83f6c60-8017-4d12-8bcc-c977483ea305'
  # segment_id = 'ca7da8ac-77c3-43c2-8ee7-9f69b0f964c4'
  file_path = f"{os.path.dirname(os.path.abspath(__file__))}/../../docs/rag/task1-glm-20241120173243-2.txt"

  # Set Knowledge Base Embedding model
  # print('--' * 50 + f"\nCall glm_dataset_setting()")
  # response = glm_dataset_setting(api_base, api_key, dataset_id)
  # pprint(response.json())

  # Create document by file
  print('--' * 50 + f"\nCall glm_document_create_by_file()")
  kwargs = {'path': file_path}
  response = glm_document_create_by_file(api_base, api_key, dataset_id, **kwargs)
  pprint(response.json())


