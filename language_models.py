import openai
import os
import time
import json
import config
from typing import Dict, List
# from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from utils import get_template
from conversation import get_conv_template

# This file is modified based on the https://github.com/YancyKahn/CoA/blob/main/language_models.py

def parse_json(self, json_str):
    json_str = json_str.replace("\n    ", "")
    json_str = json_str.replace("```json\n", "")
    json_str = json_str.replace("```", "")

    return json.loads(json_str)
    # try:
    #     json_str = json.loads(json_str)
    #     return json_str
    # except json.JSONDecodeError as e:
    #     print(f"JSON解析失败: {e}")
    #     return None

class LanguageModel():
    def __init__(self, model_name):
        self.model_name = model_name

    def batched_generate(self, prompts_list: List, max_n_tokens: int, temperature: float, top_p: float = 1.0,):
        """
        Generates responses for a batch of prompts using a language model.
        """
        raise NotImplementedError
    
    def chat(self, convs: List[Dict], max_n_tokens: int, temperature: float, top_p: float = 1.0,):
        """
        Chat with the model.
        """
        raise NotImplementedError
    

    
# class LMDeploy(LanguageModel):
#     def __init__(self, model_name):
#         self.model_name = model_name
#         backend_config = TurbomindEngineConfig(cache_max_entry_count=0.2, session_len=8000,tp=2) # 这里设置参照lmdeploy的文档
#         self.model = pipeline(self.model_name, backend_config=backend_config) # pipeline(self.model_id, backend_config=backend_config)
        
#     def batched_generate(self, convs_list: List[List[Dict]], max_n_tokens: int, temperature: float, top_p: float = 1.0,):
#         """
#         Generates responses for a batch of prompts using a language model.
#         """
#         gen_config = GenerationConfig(top_p=top_p, 
#                                         temperature=temperature,
#                                         max_new_tokens=max_n_tokens)
#         responses = self.model(convs_list, gen_config=gen_config)
#         output_list = [response.text for response in responses]
#         return output_list
        

class GPT(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 120

    def __init__(self, model_name, api_key=config.OPENAI_API_KEY) -> None:
        self.model_name = model_name
        self.api_key = api_key

        if config.IS_USE_PROXY_OPENAI:
            openai.proxy = config.PROXY

        if config.IS_USE_CUSTOM_OPENAI_API_BASE:
            self.api_base = config.OPENAI_API_BASE
            self.client = openai.OpenAI(api_key=self.api_key, base_url=self.api_base, timeout=self.API_TIMEOUT)
        else:
            self.client = openai.OpenAI(api_key=self.api_key, timeout=self.API_TIMEOUT)
        

    def generate(self, conv: List[Dict],
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                # print(response)
                output = response.choices[0].message.content

                break

            except Exception as e:  # 捕获所有异常，并记录异常信息
                print(type(e), e)
                time.sleep(self.API_RETRY_SLEEP)

            time.sleep(self.API_QUERY_SLEEP)
        return output
    
    def batched_generate(self,
                        convs_list: List[List[Dict]],
                        max_n_tokens: int,
                        temperature: float,
                        top_p: float = 1.0,
                        is_get_attention: bool = False):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]
    
class CommercialAPI(LanguageModel):
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 1
    API_MAX_RETRY = 5
    API_TIMEOUT = 500

    def __init__(self, model_name):
        self.model_name = model_name
        self.base_url = config.SF_API_BASE            
        self.api_key = config.SF_API_KEY
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        
    def generate(self, conv: List[Dict],
                            max_n_tokens: int,
                            temperature: float,
                            top_p: float,
                            is_get_attention: bool = False):
        
        output = self.API_ERROR_OUTPUT
        for _ in range(self.API_MAX_RETRY):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv,
                    max_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                output = response.choices[0].message.content
                break
            
            except Exception as e:  # 捕获所有异常，并记录异常信息
                #Handle API error here, e.g. retry or log
                print(f"OpenAI API returned an API Error: {e}")
                pass
        
        return output
    
    def batched_generate(self,
                         convs_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0,):
        return [self.generate(conv, max_n_tokens, temperature, top_p) for conv in convs_list]


if __name__ == "__main__":
    # models = ["meta-llama/Meta-Llama-3.1-70B-Instruct", "Qwen/Qwen2.5-72B-Instruct"]
    models = ["gpt-4o-mini"]

    for model in models:
        print("======="*10)
        print("Model: {}".format(model))

        lm = GPT(model)
        template = get_template(model)
        conv = get_conv_template(template)

        conv.append_message(conv.roles[0], "Hi! Please introduce yourself!")
        conv = conv.to_openai_api_messages()

        print(lm.generate(conv, 100, 1.0, 1.0))