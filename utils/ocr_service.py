import os
import base64
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv

load_dotenv()

class OCRService:
    _supabase_url = os.getenv('SUPABASE_URL', '')
    _supabase_anon_key = os.getenv('SUPABASE_ANON_KEY', '')
    _function_name = 'gcv-endpoint'

    @classmethod
    def is_configured(cls) -> bool:
        return bool(cls._supabase_url and cls._supabase_anon_key)

    @classmethod
    def extract_text_from_file(cls, image_path: str) -> str:
        if not cls.is_configured():
            raise Exception('Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file')
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            return cls.extract_text_from_bytes(image_bytes)
        except Exception as e:
            print(f'Error reading image file: {e}')
            raise Exception(f'Failed to read image file: {e}')

    @classmethod
    def extract_text_from_bytes(cls, image_bytes: bytes) -> str:
        if not cls.is_configured():
            raise Exception('Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file')
        try:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            request_body = {'image_base64': base64_image}
            function_url = f'{cls._supabase_url}/functions/v1/{cls._function_name}'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {cls._supabase_anon_key}',
            }
            response = requests.post(function_url, json=request_body, headers=headers)
            if response.status_code == 200:
                response_data = response.json()
                return cls._parse_text_from_supabase_response(response_data)
            else:
                print(f'Supabase OCR function error: {response.status_code} - {response.text}')
                raise Exception(f'OCR function request failed: {response.status_code}')
        except Exception as e:
            print(f'Error in OCR processing: {e}')
            raise Exception(f'OCR processing failed: {e}')

    @staticmethod
    def _parse_text_from_supabase_response(response: Dict[str, Any]) -> str:
        try:
            text = response.get('text')
            if text:
                return text.strip()
            error = response.get('error')
            if error:
                print(f'OCR function returned error: {error}')
                raise Exception(f'OCR function error: {error}')
            return ''
        except Exception as e:
            print(f'Error parsing Supabase OCR response: {e}')
            return ''

    @classmethod
    def extract_text_from_file_multipart(cls, image_path: str) -> str:
        if not cls.is_configured():
            raise Exception('Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file')
        try:
            function_url = f'{cls._supabase_url}/functions/v1/{cls._function_name}'
            headers = {'Authorization': f'Bearer {cls._supabase_anon_key}'}
            files = {'image': open(image_path, 'rb')}
            response = requests.post(function_url, files=files, headers=headers)
            if response.status_code == 200:
                response_data = response.json()
                return cls._parse_text_from_supabase_response(response_data)
            else:
                print(f'Supabase OCR function error: {response.status_code} - {response.text}')
                raise Exception(f'OCR function request failed: {response.status_code}')
        except Exception as e:
            print(f'Error in multipart OCR processing: {e}')
            raise Exception(f'Multipart OCR processing failed: {e}')

    @staticmethod
    def parse_receipt_data(text: str) -> Dict[str, Any]:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        receipt_data = {
            'items': [],
            'total': 0.0,
            'date': None,
            'merchant': None,
            'rawText': text,
        }
        import re
        price_pattern = re.compile(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)')
        date_pattern = re.compile(r'(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})')
        total_pattern = re.compile(r'(?:total|jumlah|grand total|subtotal)[:\s]*(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)', re.IGNORECASE)
        if lines:
            receipt_data['merchant'] = lines[0]
        for line in lines:
            date_match = date_pattern.search(line)
            if date_match:
                receipt_data['date'] = date_match.group(1)
                break
        for line in lines:
            total_match = total_pattern.search(line)
            if total_match:
                total_str = total_match.group(1).replace(',', '').replace('.', '')
                try:
                    if len(total_str) > 2:
                        total = float(total_str[:-2] + '.' + total_str[-2:])
                        receipt_data['total'] = total
                    else:
                        receipt_data['total'] = float(total_str)
                except Exception as e:
                    print(f'Error parsing total amount: {e}')
                break
        items = []
        for line in lines:
            if re.search(r'(total|subtotal|tax|pajak|merchant|toko|date|tanggal)', line, re.IGNORECASE):
                continue
            price_match = price_pattern.search(line)
            if price_match:
                price_str = price_match.group(1).replace(',', '').replace('.', '')
                try:
                    item_name = line[:price_match.start()].strip()
                    if item_name and len(item_name) > 2:
                        if len(price_str) > 2:
                            price = float(price_str[:-2] + '.' + price_str[-2:])
                        else:
                            price = float(price_str)
                        items.append({
                            'name': item_name,
                            'price': price,
                            'quantity': 1,
                        })
                except Exception as e:
                    print(f'Error parsing item: {e}')
        receipt_data['items'] = items
        return receipt_data

    @staticmethod
    def is_valid_receipt(text: str) -> bool:
        if not text.strip():
            return False
        lower_text = text.lower()
        receipt_keywords = [
            'total', 'subtotal', 'price', 'qty', 'amount',
            'receipt', 'invoice', 'bill', 'struk', 'nota'
        ]
        keyword_count = sum(1 for keyword in receipt_keywords if keyword in lower_text)
        import re
        price_pattern = re.compile(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?')
        price_matches = len(price_pattern.findall(text))
        return keyword_count >= 2 or price_matches >= 3

    @staticmethod
    def preprocess_image(image_bytes: bytes) -> bytes:
        # Placeholder for image preprocessing (contrast, noise reduction, etc.)
        return image_bytes
