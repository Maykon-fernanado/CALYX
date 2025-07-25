# CALYX: Statically Typed Prompts for Python

🔧 **Write prompts like functions. Enforce structure like Pydantic.** 🧠  
**Finally, prompt engineering you can debug, test, and trust.**

CALYX (YAML + JSON + Prompt + Human) is a type-safe prompt engineering layer that lets you:

* ✅ **Define strict input/output schemas for LLMs**
* ✅ **Validate prompt payloads like real software**  
* ✅ **Debug LLM responses with explainable logs**
* ✅ **Build modular, testable, auditable AI workflows**

Think: `TypedDict` + `Pydantic` + `LLM router` — all powered by YAML.

---

## 🚨 The Problem

**Prompt engineering is the new spaghetti code.**

* Inputs are unvalidated
* Outputs are unpredictable  
* Errors are silent
* Logic is hidden in natural language

You wouldn't ship an API like this. Why do it with AI?

---

## ✅ The Solution

**CALYX brings type safety to AI logic.**

### Define inputs/outputs in YAML:
```yaml
inputs:
  name: { type: string, min_length: 1 }
  tone: { enum: ["friendly", "formal"] }

outputs:
  greeting: { type: string }

prompt: |
  Write a {{ tone }} greeting for {{ name }}.
```

### At runtime:
```json
{
  "inputs": { "name": "Alice", "tone": "friendly" },
  "output": { "greeting": "Hey Alice!" }
}
```

**Invalid inputs or hallucinated outputs?** CALYX throws structured, explainable errors.

---

## 🏁 Quickstart

### Installation
```bash
pip install CALYX
```

### 1. Create a schema file (`greeting.yaml`):
```yaml
inputs:
  name:
    type: string
    min_length: 1
    required: true
  tone:
    type: string
    enum: ["friendly", "formal", "casual"]
    required: true

outputs:
  greeting:
    type: string
    min_length: 1

prompt: |
  Write a {{ tone }} greeting for {{ name }}.
  Return as JSON: {"greeting": "your greeting here"}
```

### 2. Use in Python:
```python
from CALYX import load_schema

# Load and execute
router = load_schema('greeting.yaml')
result = router.execute({
    'name': 'Alice',
    'tone': 'friendly'
})

print(f"Success: {result.success}")
print(f"Output: {result.output}")
```

### 3. Use from CLI:
```bash
# Validate inputs only
CALYX --schema greeting.yaml --input test.json --validate-only

# Full execution
CALYX --schema greeting.yaml --input test.json --output result.json
```

---

## 🔁 Built for Real AI Systems

### 🛠️ **Works with OpenAI, Anthropic, Local Models**

```python
from CALYX.llm_clients import CALYXRouter

router = CALYXRouter(schema_file='greeting.yaml')

# OpenAI
router.set_llm_client('openai', api_key='your-key', model='gpt-4')

# Anthropic
router.set_llm_client('anthropic', api_key='your-key', model='claude-3-sonnet')

# Local (Ollama)
router.set_llm_client('local', base_url='http://localhost:11434', model='llama2')

result = router.execute_with_llm({'name': 'Alice', 'tone': 'friendly'})
```

### 🧪 **Testable, Version-Controlled Logic**

```python
# Batch testing
test_cases = [
    {'name': 'Alice', 'tone': 'friendly'},
    {'name': 'Dr. Smith', 'tone': 'formal'},
    {'name': 'Bob', 'tone': 'casual'}
]

results = router.test_prompt(test_cases)
print(f"Passed: {sum(r.success for r in results)}/{len(results)}")
```

### 🔁 **Swappable Agents, Clean Logs**

```python
# Every execution includes full audit trail
result = router.execute(inputs)
for step in result.audit_trail:
    print(f"🔍 {step}")

# Detailed validation errors
if not result.success:
    for error in result.errors:
        print(f"❌ {error}")
```

---

## 📚 Advanced Features

### **Complex Schema Validation**
```yaml
inputs:
  user_profile:
    type: object
    properties:
      age: { type: integer, min: 13, max: 120 }
      preferences: { type: array, items: { type: string } }
  content_type:
    type: string
    enum: ["blog", "email", "tweet", "essay"]
  
outputs:
  content: { type: string, min_length: 100 }
  metadata:
    type: object
    properties:
      word_count: { type: integer }
      readability_score: { type: number }
```

### **Multi-Step Prompts**
```yaml
# Chain multiple prompts together
steps:
  - name: "analyze"
    inputs: { text: string }
    outputs: { sentiment: string, topics: array }
    prompt: "Analyze this text..."
  
  - name: "generate"  
    inputs: { sentiment: string, topics: array, style: string }
    outputs: { response: string }
    prompt: "Based on {{ sentiment }} and {{ topics }}, write..."
```

### **Conditional Logic**
```yaml
prompt: |
  {% if user_type == "expert" %}
  Provide a technical explanation of {{ topic }}.
  {% else %}
  Explain {{ topic }} in simple terms.
  {% endif %}
  
  {% if include_examples %}
  Include 2-3 practical examples.
  {% endif %}
```

---

## 💡 Why It Matters

**LLMs are too powerful to be unchecked.**

CALYX is your **type system**, **debugger**, and **reasoning layer** — in one.

### Before CALYX:
```python
# ❌ Prompt soup
prompt = f"Hey {name}, write something {tone} about {topic}"
response = llm.generate(prompt)  # 🤞 Hope it works
result = json.loads(response)    # 💥 Often breaks
```

### After CALYX:
```python
# ✅ Typed, validated, auditable
router = load_schema('content_generator.yaml')
result = router.execute({
    'name': name,
    'tone': tone, 
    'topic': topic
})
# Guaranteed valid input/output or structured error
```

### Benefits:
* 🛡️ **No more prompt soup**
* 🔧 **No more JSON schema duct tape** 
* 🚫 **No more hallucinated pipelines**
* 📋 **Built-in audit trails**
* 🧪 **Unit test your prompts**
* 📊 **Performance monitoring**
* 🔒 **Compliance ready**

---

## 🔮 What's Next?

CALYX is just the beginning. Coming soon:

### 🧠 **Universal Reasoning Bus**
Multi-agent coordination with typed message passing:
```yaml
agents:
  researcher: { schema: research.yaml }
  writer: { schema: writing.yaml }
  editor: { schema: editing.yaml }

workflow:
  - researcher -> writer: { findings: array, sources: array }
  - writer -> editor: { draft: string, metadata: object }
```

### 📊 **Monte Carlo Prompt Simulations**
Test prompt reliability across hundreds of variations:
```python
simulation = router.monte_carlo_test(
    input_variations=1000,
    success_threshold=0.95
)
```

### 🔒 **Permissioned Cognitive Graphs**
Fine-grained access control for AI decision-making:
```yaml
permissions:
  finance_team: ["loan_approval", "risk_assessment"]  
  hr_team: ["resume_screening", "interview_scheduling"]
```

---

## 🧠 Philosophy

Under the hood, CALYX is a new kind of **reasoning infrastructure**.

We treat **YAML as the cognitive OS** — and **LLMs as modular workers**.

It's not just about prompts. It's about **trustable, auditable thinking at scale**.

But you don't need to believe that yet.  
Just try typing your first prompt like it's a function — and see what happens.

---

## 📦 Full API Reference

### Core Classes

#### `Router`
```python
router = Router(schema_file='schema.yaml')
router = Router(schema_dict={...})

# Validate inputs only
result = router.validate_inputs(data)

# Render prompt template  
prompt = router.render_prompt(data)

# Full execution (validation + rendering)
result = router.execute(data)

# Batch testing
results = router.test_prompt(test_cases)
```

#### `PromptResult`
```python
result.success          # bool: Overall success
result.output           # dict: Parsed LLM output  
result.errors           # list: Validation errors
result.prompt_used      # str: Rendered prompt
result.execution_time   # float: Total time
result.audit_trail      # list: Step-by-step log
```

### LLM Integration

#### `CALYXRouter` 
```python
from CALYX.llm_clients import CALYXRouter

router = CALYXRouter(schema_file='schema.yaml')
router.set_llm_client('openai', api_key='...', model='gpt-4')
result = router.execute_with_llm(inputs)
```

#### Supported Providers
* **OpenAI**: `gpt-4`, `gpt-3.5-turbo`, etc.
* **Anthropic**: `claude-3-sonnet`, `claude-3-haiku`, etc.  
* **Local**: Ollama, LM Studio, vLLM
* **Mock**: For testing and development

### Schema Format

#### Input Schema
```yaml
inputs:
  field_name:
    type: string|integer|number|boolean|array|object
    required: true|false
    min_length: 1        # for strings
    max_length: 100      # for strings  
    min: 0              # for numbers
    max: 100            # for numbers
    enum: ["opt1", "opt2"]  # allowed values
    default: "value"     # default value
    description: "Field description"

outputs:
  field_name:
    type: string|integer|number|boolean|array|object
    required: true|false
    # Same validation options as inputs
```

#### Full Schema Example
```yaml
# Metadata
metadata:
  name: "Content Generator"
  version: "1.0.0"
  description: "Generates content with specified tone and style"
  author: "Your Team"
  tags: ["content", "generation", "marketing"]

# Input validation
inputs:
  topic:
    type: string
    min_length: 3
    max_length: 100
    required: true
    description: "The main topic to write about"
  
  audience:
    type: string
    enum: ["beginners", "professionals", "experts"]
    required: true
    description: "Target audience level"
  
  word_count:
    type: integer
    min: 100
    max: 2000
    default: 500
    description: "Desired word count"
  
  include_examples:
    type: boolean
    default: false
    description: "Whether to include examples"

# Output validation  
outputs:
  title:
    type: string
    min_length: 5
    max_length: 100
    required: true
  
  content:
    type: string
    min_length: 50
    required: true
  
  metadata:
    type: object
    properties:
      word_count: { type: integer }
      reading_time: { type: integer }
      complexity_score: { type: number, min: 0, max: 1 }

# Prompt template
prompt: |
  Write a comprehensive article about {{ topic }} for {{ audience }}.
  
  Requirements:
  - Target length: ~{{ word_count }} words
  - Audience: {{ audience }}
  {% if include_examples %}
  - Include 2-3 practical examples
  {% endif %}
  
  Return your response as JSON:
  {
    "title": "engaging article title",
    "content": "full article content",
    "metadata": {
      "word_count": estimated_words,
      "reading_time": estimated_minutes,
      "complexity_score": 0.0-1.0
    }
  }

# Model configuration
model:
  provider: "openai"
  name: "gpt-4"
  temperature: 0.7
  max_tokens: 2000
  top_p: 0.9

# Validation settings
validation:
  strict_mode: true
  allow_extra_outputs: false
  timeout_seconds: 30
```

---

## 🛠️ CLI Usage

### Basic Commands
```bash
# Validate input against schema
CALYX --schema schema.yaml --input data.json --validate-only

# Execute with LLM (requires API keys)
CALYX --schema schema.yaml --input data.json --output result.json

# Test multiple inputs
CALYX --schema schema.yaml --input test_cases.json --batch

# Debug mode with full audit trail
CALYX --schema schema.yaml --input data.json --debug
```

### Environment Variables
```bash
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export CALYX_DEFAULT_PROVIDER="openai"
export CALYX_DEFAULT_MODEL="gpt-4"
```

### Configuration File
```yaml
# ~/.CALYX/config.yaml
default_provider: "openai"
default_model: "gpt-4"
timeout: 30
cache_enabled: true
log_level: "INFO"

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"
  
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
  
  local:
    base_url: "http://localhost:11434"
```

---

## 🧪 Testing & Development

### Unit Testing Your Prompts
```python
import pytest
from CALYX import load_schema

@pytest.fixture
def content_router():
    return load_schema('schemas/content_generator.yaml')

def test_valid_inputs(content_router):
    """Test that valid inputs pass validation"""
    valid_input = {
        'topic': 'Machine Learning',
        'audience': 'beginners',
        'word_count': 800
    }
    
    result = content_router.validate_inputs(valid_input)
    assert result.success
    assert len(result.errors) == 0

def test_invalid_topic_length(content_router):
    """Test that short topics are rejected"""
    invalid_input = {
        'topic': 'AI',  # too short
        'audience': 'beginners',
        'word_count': 500
    }
    
    result = content_router.validate_inputs(invalid_input)
    assert not result.success
    assert any('min_length' in error for error in result.errors)

def test_prompt_rendering(content_router):
    """Test prompt template rendering"""
    inputs = {
        'topic': 'Python Programming',
        'audience': 'professionals',
        'word_count': 1200,
        'include_examples': True
    }
    
    prompt = content_router.render_prompt(inputs)
    assert 'Python Programming' in prompt
    assert 'professionals' in prompt
    assert '1200' in prompt
    assert 'practical examples' in prompt

def test_batch_execution(content_router):
    """Test multiple inputs at once"""
    test_cases = [
        {'topic': 'DevOps', 'audience': 'professionals', 'word_count': 600},
        {'topic': 'Web Design', 'audience': 'beginners', 'word_count': 400},
        {'topic': 'Data Science', 'audience': 'experts', 'word_count': 1000}
    ]
    
    results = content_router.test_prompt(test_cases)
    assert len(results) == 3
    assert all(r.input_validation['valid'] for r in results)
```

### Integration Testing with Mock LLM
```python
from CALYX.llm_clients import CALYXRouter, MockLLMClient

def test_end_to_end_execution():
    """Test full pipeline with mock LLM"""
    router = CALYXRouter(schema_file='schemas/greeting.yaml')
    router.llm_client = MockLLMClient()
    
    result = router.execute_with_llm({
        'name': 'Alice',
        'tone': 'friendly'
    })
    
    assert result['success']
    assert 'greeting' in result['output']
    assert result['llm_metadata']['model'] == 'mock-gpt'
```

### Performance Testing
```python
import time
from CALYX import load_schema

def test_prompt_performance():
    """Test prompt execution performance"""
    router = load_schema('schemas/fast_prompt.yaml')
    
    start_time = time.time()
    
    # Test 100 executions
    for i in range(100):
        result = router.validate_inputs({'text': f'test input {i}'})
        assert result.success
    
    end_time = time.time()
    avg_time = (end_time - start_time) / 100
    
    # Should validate inputs in under 1ms each
    assert avg_time < 0.001
```

---

## 🚀 Production Deployment

### Docker Container
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install CALYX
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy schemas and app
COPY schemas/ ./schemas/
COPY app.py .

# Environment variables
ENV CALYX_SCHEMA_DIR=/app/schemas
ENV CALYX_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "from CALYX import load_schema; load_schema('schemas/health.yaml')"

CMD ["python", "app.py"]
```

### FastAPI Integration
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from CALYX.llm_clients import CALYXRouter
import os

app = FastAPI(title="CALYX API Service")

# Load routers at startup
routers = {
    'greeting': CALYXRouter(schema_file='schemas/greeting.yaml'),
    'content': CALYXRouter(schema_file='schemas/content.yaml'),
    'analysis': CALYXRouter(schema_file='schemas/analysis.yaml')
}

# Configure LLM clients
for router in routers.values():
    router.set_llm_client(
        'openai',
        api_key=os.getenv('OPENAI_API_KEY'),
        model='gpt-4'
    )

class PromptRequest(BaseModel):
    schema_name: str
    inputs: dict

@app.post("/execute")
async def execute_prompt(request: PromptRequest):
    """Execute a CALYX prompt"""
    
    if request.schema_name not in routers:
        raise HTTPException(404, f"Schema '{request.schema_name}' not found")
    
    router = routers[request.schema_name]
    result = router.execute_with_llm(request.inputs)
    
    if not result['success']:
        raise HTTPException(400, {
            'errors': result['errors'],
            'stage': result.get('stage', 'unknown')
        })
    
    return {
        'output': result['output'],
        'metadata': result.get('llm_metadata', {}),
        'execution_time': result.get('execution_time', 0)
    }

@app.get("/schemas")
async def list_schemas():
    """List available schemas"""
    return {'schemas': list(routers.keys())}

@app.post("/validate")
async def validate_inputs(request: PromptRequest):
    """Validate inputs without calling LLM"""
    
    if request.schema_name not in routers:
        raise HTTPException(404, f"Schema '{request.schema_name}' not found")
    
    router = routers[request.schema_name]
    result = router.router.validate_inputs(request.inputs)
    
    return {
        'valid': result.success,
        'errors': result.errors
    }
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: CALYX-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: CALYX-service
  template:
    metadata:
      labels:
        app: CALYX-service
    spec:
      containers:
      - name: CALYX
        image: your-registry/CALYX-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: openai-key
        - name: ANTHROPIC_API_KEY
          valueFrom:
            secretKeyRef:
              name: llm-secrets
              key: anthropic-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: CALYX-service
spec:
  selector:
    app: CALYX-service
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## 🙌 Who's It For?

### 👨💻 **Engineers building serious LLM apps**
- Type-safe prompt APIs
- Unit testable AI logic  
- Production-ready error handling

### 🔧 **DevOps debugging brittle AI pipelines**
- Structured validation errors
- Complete audit trails
- Performance monitoring

### 🤖 **Agents routing AI logic with context**
- Schema-validated message passing
- Multi-step reasoning workflows
- Modular prompt composition

### 😤 **Anyone tired of praying to the prompt gods**
- No more "it works on my machine"
- Reproducible AI behaviors
- Debuggable failures

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/CALYX.git
cd CALYX

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev,llm]"

# Run tests
pytest tests/

# Format code
black CALYX/
isort CALYX/

# Lint
flake8 CALYX/
```

### Project Structure
```
CALYX/
├── CALYX/
│   ├── __init__.py          # Core Router class
│   ├── llm_clients.py       # LLM integrations
│   ├── validators.py        # Schema validation
│   └── templates.py         # Prompt rendering
├── examples/
│   ├── greeting.yaml        # Example schemas
│   ├── content_gen.yaml
│   └── test_cases.json
├── tests/
│   ├── test_validation.py   # Unit tests
│   ├── test_rendering.py
│   └── test_integration.py
├── docs/                    # Documentation
├── setup.py                 # Package config
└── README.md               # This file
```

### Contribution Guidelines
1. **Issues**: Use GitHub issues for bugs and feature requests
2. **Pull Requests**: Follow the PR template and ensure tests pass
3. **Code Style**: Use Black + isort for formatting
4. **Tests**: Add tests for new features
5. **Documentation**: Update docs for user-facing changes

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 💬 Community & Support

### 🔗 **Links**
- **GitHub**: https://github.com/yourusername/CALYX
- **Documentation**: https://CALYX.readthedocs.io
- **PyPI**: https://pypi.org/project/CALYX
- **Discussions**: https://github.com/yourusername/CALYX/discussions

### 🐛 **Bug Reports**
Found a bug? Please open an issue with:
- Python version
- CALYX version  
- Minimal reproduction case
- Expected vs actual behavior

### 💡 **Feature Requests**
Have ideas? We'd love to hear them! Open an issue with:
- Use case description
- Proposed API design
- Why existing features don't solve it

### 🆘 **Getting Help**
- Check the [docs](https://CALYX.readthedocs.io) first
- Search [existing issues](https://github.com/yourusername/CALYX/issues)
- Ask in [Discussions](https://github.com/yourusername/CALYX/discussions)
- For urgent issues, email: support@CALYX.dev

---

## 🎉 Final Words

**CALYX is more than a library — it's a philosophy.**

We believe AI systems should be:
- **Predictable** (typed inputs/outputs)
- **Debuggable** (structured errors)  
- **Testable** (unit test your prompts)
- **Auditable** (complete execution logs)
- **Maintainable** (version-controlled logic)

Stop praying to the prompt gods. Start building cognitive software like real software.

**Ready to get started?**

```bash
pip install CALYX
```

```python
from CALYX import create_router

router = create_router(
    inputs={'name': {'type': 'string', 'required': True}},
    outputs={'greeting': {'type': 'string'}},
    prompt='Hello {{ name }}! Welcome to typed prompts.'
)

result = router.execute({'name': 'World'})
print(f"Success: {result.success}")
```

---

⭐ **Star this repo if you believe AI systems should explain themselves**  
🚀 **Join us in building the future of trustable AI**

---

*CALYX: Finally, prompt engineering you can debug, test, and trust.*
