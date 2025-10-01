import os
import json
import requests
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class OpenAIResult:
    def __init__(self, content: str, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None,
                 total_tokens: Optional[int] = None, reasoning_tokens: Optional[int] = None, model: str = ''):
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.reasoning_tokens = reasoning_tokens
        self.model = model

    def __str__(self):
        return (f'OpenAIResult(content: {len(self.content)} chars, '
                f'promptTokens: {self.prompt_tokens}, '
                f'completionTokens: {self.completion_tokens}, '
                f'totalTokens: {self.total_tokens}, '
                f'reasoningTokens: {self.reasoning_tokens}, '
                f'model: {self.model})')

class OpenAIService:
    max_input_tokens = 8192
    max_output_tokens = 4096

    def __init__(self):
        self._chat_history: List[Dict[str, str]] = []
        self._system_prompt: Optional[str] = None

    @staticmethod
    def _truncate_to_max_tokens(text: str, max_tokens: int) -> str:
        max_chars = max_tokens * 4
        return text[:max_chars] if len(text) > max_chars else text

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def add_message_to_history(self, role: str, content: str):
        self._chat_history.append({'role': role, 'content': content})

    def clear_history(self):
        self._chat_history.clear()

    def get_chat_history(self) -> List[Dict[str, str]]:
        return list(self._chat_history)

    def get_system_prompt(self) -> Optional[str]:
        return self._system_prompt

    def has_history(self) -> bool:
        return bool(self._chat_history)

    def _build_prompt(self, user_message: str) -> str:
        prompt_parts = []
        if self._system_prompt:
            prompt_parts.append(f'System: {self._system_prompt}')
        for message in self._chat_history:
            role = message['role'].capitalize()
            content = message['content']
            prompt_parts.append(f'{role}: {content}')
        truncated_message = self._truncate_to_max_tokens(user_message, self.max_input_tokens)
        prompt_parts.append(f'User: {truncated_message}')
        prompt_parts.append('Assistant:')
        return '\n\n'.join(prompt_parts)

    @property
    def is_configured(self) -> bool:
        return bool(self._supabase_url and self._supabase_anon_key)

    @property
    def _supabase_url(self) -> str:
        return os.getenv('SUPABASE_URL', '')

    @property
    def _supabase_anon_key(self) -> str:
        return os.getenv('SUPABASE_ANON_KEY', '')

    _function_name = 'openai-gpt-function'

    def send_message(self, message: str, model: str = 'gpt-5-mini', max_tokens: Optional[int] = None) -> str:
        result = self.send_message_with_tokens(message, model=model, max_tokens=max_tokens)
        return result.content

    def send_message_with_tokens(self, message: str, model: str = 'gpt-5-mini', max_tokens: Optional[int] = None) -> OpenAIResult:
        if not self.is_configured:
            raise Exception('Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file')
        function_url = f'{self._supabase_url}/functions/v1/{self._function_name}'
        prompt = self._build_prompt(message)
        request_body = {'prompt': prompt}
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self._supabase_anon_key}',
        }
        response = requests.post(function_url, headers=headers, data=json.dumps(request_body))
        if response.status_code == 200:
            data = response.json()
            text_field = data.get('text')
            assistant_response = text_field if isinstance(text_field, str) else json.dumps(text_field)
            tokens_data = data.get('tokens')
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            reasoning_tokens = None
            if tokens_data:
                prompt_tokens = tokens_data.get('prompt_tokens') or tokens_data.get('input')
                completion_tokens = tokens_data.get('completion_tokens') or tokens_data.get('output')
                total_tokens = tokens_data.get('total_tokens') or tokens_data.get('total')
                reasoning_tokens = tokens_data.get('reasoning')
            model_used = data.get('model', model)
            self.add_message_to_history('user', message)
            self.add_message_to_history('assistant', assistant_response)
            return OpenAIResult(
                content=assistant_response,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                model=model_used
            )
        else:
            raise Exception(f'Failed to get response from Supabase function: {response.text}')

    def stream_chat_message(self, message: str, model: str = 'gpt-5-mini', max_tokens: Optional[int] = None):
        import time
        if not self.is_configured:
            raise Exception('Supabase configuration not found. Please set SUPABASE_URL and SUPABASE_ANON_KEY in .env file')
        result = self.send_message_with_tokens(message, model=model, max_tokens=max_tokens)
        chunk_size = 10
        response = result.content
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i+chunk_size]
            yield chunk
            time.sleep(0.05)
