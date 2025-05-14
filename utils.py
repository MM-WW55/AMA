from conversation import get_conv_template, register_conv_template, \
    Conversation, conv_templates, SeparatorStyle
import config
import ast
from loggers import logger

def extract_json(s):
    """
    Given an output from the attacker LLM, this function extracts the values
    for `improvement` and `adversarial prompt` and returns them as a dictionary.

    Args:
        s (str): The string containing the potential JSON structure.

    Returns:
        dict: A dictionary containing the extracted values.
        str: The cleaned JSON string.
    """
    # Extract the string that looks like a JSON
    start_pos = s.find("{") 
    end_pos = s.rfind("}") +1  # +1 to include the closing brace
    if end_pos == -1:
        logger.error("Error extracting potential JSON structure")
        logger.error(f"Input:\n {s}")
        return None, None

    json_str = s[start_pos:end_pos]
    json_str = json_str.replace("\n    ", "")
    json_str = json_str.replace("\n", "")  # Remove all line breaks

    try:
        parsed = ast.literal_eval(json_str)
        if not all(x in parsed for x in ["improvement","prompt"]):
            logger.error("Error in extracted structure. Missing keys.")
            logger.error(f"Extracted:\n {json_str}")
            return None, None
        return parsed, json_str
    except (SyntaxError, ValueError):
        logger.error("Error parsing extracted structure")
        logger.error(f"Extracted:\n {json_str}")
        return None, None

def get_template(model_name: str):
    model_name = config.MODELPOOL[model_name]
    if "qwen" in model_name:
        return "qwen-7b-chat"
    elif "vicuna" in model_name:
        return "vicuna_v1.1"
    elif "deepseek" in model_name:
        return "deepseek-chat"
    else:
        return "api_based_default"

if __name__ == "__main__":

    conv = get_conv_template("vicuna_v1.1")
    conv.append_message(conv.roles[0], "Hello!")
    conv.append_message(conv.roles[1], "Hi!")
    conv.append_message(conv.roles[0], "How are you?")
    conv.append_message(conv.roles[1], None)

    print(conv)
    print("\n\n")
    print(conv.to_openai_api_messages())
