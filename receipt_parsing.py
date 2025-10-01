import pandas as pd
from ocr_service import OCRService
from openai_service import OpenAIService
from prompt_receipt_parsing import create_receipt_parsing_prompt

openai_service = OpenAIService()
image_path_1 = 'assets/images/image_1.jpeg'

def receipt_parsing_with_openai(image_path: str):
    ocr_result = OCRService.extract_text_from_file(image_path)
    try:
        prompt = create_receipt_parsing_prompt(ocr_result)
        parsed_result = openai_service.send_message_with_tokens(
            prompt,
            model="gpt-5-mini"
        )
        print('OpenAI Parsed Result:', parsed_result)
    except Exception as e:
        print('Error using OpenAIService:', e)