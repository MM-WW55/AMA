import os

# Set your own OpenAI API key

OPENAI_API_KEY="PUT-YOUR-KEY_HERE"
OPENAI_API_BASE=""

# Set your own SilianFlow API key
SF_API_KEY=""
SF_API_BASE=""

IS_USE_CUSTOM_OPENAI_API_BASE = True
IS_USE_PROXY_OPENAI = False

PROXY = {
    "http": "http://localhost:7890",
    "https": "http://localhost:7890",
}

# Experiment parameters setting
ATTACK_TEMP = 1
TARGET_TEMP = 0
ATTACK_TOP_P = 0.9
TARGET_TOP_P = 1
JUDGE_TEMP = 1
JUDGE_TOP_P = 0.9
MAX_ATTACK_ROUNDs = 3

MODELPOOL = {
    "/root/repos/CoA/vicuna-13b-v1.5-16k": "vicuna1.5_13b"}

APIPOOL = {}
