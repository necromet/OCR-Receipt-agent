from typing import Optional

from utils.ocr_service import OCRService
from utils.openai_service import OpenAIResult, OpenAIService
from utils.prompt_receipt_parsing import create_receipt_parsing_prompt


def _parse_receipt_text_with_openai(ocr_text: str, *, model: str = "gpt-5-mini") -> OpenAIResult:
    """Send OCR text through OpenAI to obtain structured receipt data."""
    openai_service = OpenAIService()
    prompt = create_receipt_parsing_prompt(ocr_text)
    return openai_service.send_message_with_tokens(prompt, model=model)


def receipt_parsing_with_openai(image_path: str, *, model: str = "gpt-5-mini") -> Optional[OpenAIResult]:
    """Parse a receipt image located on disk and return the OpenAI result."""
    try:
        ocr_result = OCRService.extract_text_from_file(image_path)
        parsed_result = _parse_receipt_text_with_openai(ocr_result, model=model)
        print('OpenAI Parsed Result:', parsed_result)
        return parsed_result
    except Exception as e:
        print('Error using OpenAIService:', e)
        return None


def receipt_parsing_from_bytes(image_bytes: bytes, *, model: str = "gpt-5-mini") -> Optional[OpenAIResult]:
    """Parse a receipt image provided as raw bytes."""
    try:
        ocr_result = OCRService.extract_text_from_bytes(image_bytes)
        return _parse_receipt_text_with_openai(ocr_result, model=model)
    except Exception as e:
        print('Error using OpenAIService:', e)
        return None
