"""
CALYX LLM Client Integrations
Support for OpenAI, Anthropic, and local models
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency: Optional[float] = None
    metadata: Dict[str, Any] = None


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, api_key: str = None, model: str = None):
        self.api_key = api_key
        self.model = model
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response from prompt"""
        pass
    
    def parse_json_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues"""
        # Try to extract JSON from markdown code blocks
        content = content.strip()
        
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Try to find JSON object in the text
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end != 0:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass
            
            raise ValueError(f"Could not parse JSON from LLM response: {e}")


class OpenAIClient(BaseLLMClient):
    """OpenAI API client for CALYX"""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        super().__init__(api_key, model)
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get('temperature', 0.7),
                max_tokens=kwargs.get('max_tokens', 1000),
                **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                model=response.model,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency=latency,
                metadata={
                    'finish_reason': response.choices[0].finish_reason,
                    'prompt_tokens': response.usage.prompt_tokens if response.usage else None,
                    'completion_tokens': response.usage.completion_tokens if response.usage else None,
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client for CALYX"""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        super().__init__(api_key, model)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API"""
        start_time = time.time()
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get('max_tokens', 1000),
                temperature=kwargs.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                model=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency=latency,
                metadata={
                    'stop_reason': response.stop_reason,
                    'input_tokens': response.usage.input_tokens,
                    'output_tokens': response.usage.output_tokens,
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {e}")


class LocalLLMClient(BaseLLMClient):
    """Local LLM client (e.g., Ollama, LM Studio)"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        super().__init__(model=model)
        self.base_url = base_url.rstrip('/')
        
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using local LLM API"""
        import requests
        
        start_time = time.time()
        
        try:
            # Ollama API format
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get('temperature', 0.7),
                        "num_predict": kwargs.get('max_tokens', 1000),
                    }
                }
            )
            
            response.raise_for_status()
            data = response.json()
            
            latency = time.time() - start_time
            
            return LLMResponse(
                content=data.get('response', ''),
                model=self.model,
                latency=latency,
                metadata={
                    'eval_count': data.get('eval_count'),
                    'eval_duration': data.get('eval_duration'),
                }
            )
            
        except Exception as e:
            raise RuntimeError(f"Local LLM API error: {e}")


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing"""
    
    def __init__(self, model: str = "mock-gpt"):
        super().__init__(model=model)
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate mock response for testing"""
        time.sleep(0.1)  # Simulate API latency
        
        # Generate a simple mock response based on prompt
        if "greeting" in prompt.lower():
            mock_content = '''```json
{
  "greeting": "Hello! Nice to meet you!",
  "sentiment_score": 0.85,
  "language_detected": "english"
}
```'''
        elif "code" in prompt.lower():
            mock_content = '''```json
{
  "overall_score": 7,
  "issues_found": ["Missing docstring", "Variable naming could be improved"],
  "suggestions": "Add documentation and use more descriptive variable names",
  "approved": true
}
```'''
        else:
            mock_content = '''```json
{
  "response": "This is a mock response for testing purposes.",
  "confidence": 0.9
}
```'''
        
        return LLMResponse(
            content=mock_content,
            model=self.model,
            tokens_used=50,
            latency=0.1,
            metadata={'mock': True}
        )


def create_llm_client(provider: str, **kwargs) -> BaseLLMClient:
    """Factory function to create LLM clients"""
    
    if provider == "openai":
        return OpenAIClient(
            api_key=kwargs.get('api_key'),
            model=kwargs.get('model', 'gpt-4')
        )
    elif provider == "anthropic":
        return AnthropicClient(
            api_key=kwargs.get('api_key'),
            model=kwargs.get('model', 'claude-3-sonnet-20240229')
        )
    elif provider == "local":
        return LocalLLMClient(
            base_url=kwargs.get('base_url', 'http://localhost:11434'),
            model=kwargs.get('model', 'llama2')
        )
    elif provider == "mock":
        return MockLLMClient(
            model=kwargs.get('model', 'mock-gpt')
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


# Integration with main Router class
class CALYXRouter:
    """Enhanced Router with LLM integration"""
    
    def __init__(self, schema_file: str = None, schema_dict: Dict[str, Any] = None):
        from CALYX import Router
        self.router = Router(schema_file, schema_dict)
        self.llm_client = None
    
    def set_llm_client(self, provider: str, **kwargs):
        """Set LLM client for this router"""
        self.llm_client = create_llm_client(provider, **kwargs)
    
    def execute_with_llm(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute prompt with LLM and return structured result"""
        from CALYX import PromptResult
        import time
        
        start_time = time.time()
        
        # Validate inputs
        validation_result = self.router.validate_inputs(inputs)
        if not validation_result.success:
            return {
                'success': False,
                'errors': validation_result.errors,
                'stage': 'input_validation'
            }
        
        # Render prompt
        try:
            rendered_prompt = self.router.render_prompt(inputs)
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Prompt rendering failed: {e}"],
                'stage': 'prompt_rendering'
            }
        
        # Call LLM
        if not self.llm_client:
            return {
                'success': False,
                'errors': ['No LLM client configured'],
                'stage': 'llm_configuration'
            }
        
        try:
            llm_response = self.llm_client.generate(
                rendered_prompt,
                **self.router.schema.get('model', {})
            )
        except Exception as e:
            return {
                'success': False,
                'errors': [f"LLM call failed: {e}"],
                'stage': 'llm_call',
                'prompt_used': rendered_prompt
            }
        
        # Parse JSON response
        try:
            parsed_output = self.llm_client.parse_json_response(llm_response.content)
        except Exception as e:
            return {
                'success': False,
                'errors': [f"JSON parsing failed: {e}"],
                'stage': 'output_parsing',
                'raw_response': llm_response.content,
                'prompt_used': rendered_prompt
            }
        
        # Validate outputs
        output_valid, output_errors = self.router.validate_outputs(parsed_output)
        if not output_valid:
            return {
                'success': False,
                'errors': output_errors,
                'stage': 'output_validation',
                'raw_output': parsed_output,
                'prompt_used': rendered_prompt
            }
        
        # Success!
        return {
            'success': True,
            'output': parsed_output,
            'prompt_used': rendered_prompt,
            'llm_metadata': {
                'model': llm_response.model,
                'tokens_used': llm_response.tokens_used,
                'latency': llm_response.latency,
                'cost': llm_response.cost
            },
            'execution_time': time.time() - start_time
        }


# Example usage function
def demo_llm_integration():
    """Demonstration of LLM integration"""
    
    # Create router with schema
    schema = {
        'inputs': {
            'topic': {'type': 'string', 'required': True},
            'tone': {'type': 'string', 'enum': ['casual', 'formal'], 'required': True}
        },
        'outputs': {
            'title': {'type': 'string', 'required': True},
            'content': {'type': 'string', 'required': True}
        },
        'prompt': 'Write a {{ tone }} blog post about {{ topic }}. Return as JSON with "title" and "content" fields.',
        'model': {
            'temperature': 0.8,
            'max_tokens': 500
        }
    }
    
    router = CALYXRouter(schema_dict=schema)
    
    # Test with mock client
    router.set_llm_client('mock')
    
    result = router.execute_with_llm({
        'topic': 'AI Safety',
        'tone': 'formal'
    })
    
    print("LLM Integration Demo Result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo_llm_integration()
