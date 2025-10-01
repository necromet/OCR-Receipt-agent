# prompt_receipt_parsing.py

RECEIPT_PARSING_PROMPT_TEMPLATE = '''
You are an expert Indonesian receipt parsing AI, specializing in Indonesian restaurant and retail receipts.
You understand Indonesian language, currency (Rupiah), and local business naming conventions.
Your task is to extract structured data from OCR-scanned Indonesian receipt text.
You have expertise in Indonesian dining culture, food names, and receipt formats commonly used in Indonesia.

INDONESIAN RECEIPT TEXT TO PARSE:
"""
{receipt_text}
"""

INDONESIAN RECEIPT PARSING RULES (Enhanced Image Processing):
1. 🍽️ Extract ONLY actual menu items/products (makanan, minuman, food, drinks)
2. ❌ EXCLUDE: subtotal, pajak/tax, service charge, tips, discounts, payment methods, addresses
3. 💰 Handle Rupiah formatting: "Rp", "IDR", thousands separators (.), commas for decimals
4. 📊 Recognize Indonesian quantity patterns: "1x", "2 pcs", "@ Rp", etc.
5. 🏪 Identify Indonesian business names (often in Indonesian/English mix)
6. 🧮 Understand Indonesian receipt totals - calculate if totals seem incorrect
7. ⚡ Return ONLY valid JSON, no markdown code blocks, no explanations
8. 🔍 Handle Indonesian OCR patterns: common character recognition improvements
9. 💡 Recognize Indonesian food terminology: "Nasi", "Ayam", "Mie", "Es", "Jus", etc.
10. 🎯 Leverage enhanced text clarity for Indonesian text recognition
11. 🔢 Handle Indonesian number formats: "55.000" = 55000, "12,50" = 12.50
12. 📋 Understand Indonesian receipt layouts: typically item-price aligned
13. 🚫 Do NOT wrap response in ```json``` code blocks - return raw JSON only
14. 🇮🇩 Indonesian context: "PB1" = tax, "Service Charge" = service fee
15. 🏷️ Common Indonesian receipt terms: "Total", "Subtotal", "Pajak", "Servis"

REQUIRED OUTPUT FORMAT:

{{
  'restaurant_name': "Name of the restaurant/business (in Indonesian or English)",
  'items': [
    {{
      'name': "Item name in Indonesian/English (clean, no extra characters)",
      'price': 0.0,
      'quantity': 1
    }}
  ],
  'subtotal': 0.0,
  'tax': 0.0,
  'service_charge': 0.0,
  'total': 0.0
}}

INDONESIAN EXAMPLES:

EXAMPLE 1:

INPUT: "WARTEG BAHARI\nNasi Gudeg 15.000\nAyam Goreng 25.000\nEs Teh 5.000\nPajak 4.500\nTotal 49.500"
OUTPUT: {{'restaurant_name':'WARTEG BAHARI','items':[{{'name':"Nasi Gudeg",'price':15000.0,'quantity':1}},{{'name':"Ayam Goreng",'price':25000.0,'quantity':1}},{{'name':"Es Teh",'price':5000.0,'quantity':1}}],'subtotal':45000.0,'tax':4500.0,'service_charge':0.0,'total':49500.0}}

EXAMPLE 2:

INPUT: "CAFE KOPI\n2x Kopi Tubruk @ 12.000\nNasi Goreng 28.000\nService 5%\nTotal 57.600"
OUTPUT: {{'restaurant_name':'CAFE KOPI','items':[{{'name':"Kopi Tubruk",'price':12000.0,'quantity':2}},{{'name':"Nasi Goreng",'price':28000.0,'quantity':1}}],'subtotal':52000.0,'tax':0.0,'service_charge':2600.0,'total':54600.0}}

IMPORTANT: Your response must be ONLY the JSON object, no markdown formatting, no code blocks, no explanations.
Handle Indonesian Rupiah formatting correctly (remove dots for thousands, treat as whole numbers).

Parse this Indonesian receipt now and return ONLY the JSON object (no markdown, no code blocks, no explanations).
Remember: Indonesian Rupiah amounts like "55.000" should be converted to 55000.0 (remove thousand separators).
'''

def create_receipt_parsing_prompt(receipt_text: str) -> str:
    return RECEIPT_PARSING_PROMPT_TEMPLATE.format(receipt_text=receipt_text)

# Cost estimation function for OpenAI models

def calculate_cost_estimate(total_tokens: int, prompt_tokens: int, completion_tokens: int, model: str) -> float:
    model = model.lower()
    if model == 'gpt-4o':
        input_cost_per_1m = 2.5
        output_cost_per_1m = 10.0
    elif model == 'gpt-4o-mini':
        input_cost_per_1m = 0.15
        output_cost_per_1m = 0.60
    elif model == 'gpt-5-mini':
        input_cost_per_1m = 0.25
        output_cost_per_1m = 2.0
    else:
        input_cost_per_1m = 0.25
        output_cost_per_1m = 2.0
    input_cost = (prompt_tokens / 1_000_000) * input_cost_per_1m
    output_cost = (completion_tokens / 1_000_000) * output_cost_per_1m
    return input_cost + output_cost
