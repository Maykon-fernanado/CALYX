#!/usr/bin/env python3
"""
CALYX Demo: Statically Typed Prompts in Action
🔧 Write prompts like functions. Enforce structure like Pydantic.
"""

from calyx import Router, create_router, PromptResult
import json


def demo_greeting_prompt():
    """Demo 1: Simple greeting with input validation"""
    print("👋 GREETING PROMPT DEMO")
    print("=" * 50)
    
    # Create a typed prompt schema
    greeting_router = create_router(
        inputs={
            'name': {'type': 'string', 'min_length': 1, 'required': True},
            'tone': {'type': 'string', 'enum': ['friendly', 'formal', 'casual'], 'required': True}
        },
        outputs={
            'greeting': {'type': 'string', 'min_length': 1}
        },
        prompt="Write a {{ tone }} greeting for {{ name }}. Be warm and appropriate."
    )
    
    # Test valid input
    print("✅ Valid Input Test:")
    valid_input = {'name': 'Alice', 'tone': 'friendly'}
    result = greeting_router.execute(valid_input)
    
    print(f"Success: {result.success}")
    print(f"Rendered Prompt: {result.prompt_used}")
    print(f"Validation: {result.input_validation}")
    print()
    
    # Test invalid input
    print("❌ Invalid Input Test:")
    invalid_input = {'name': '', 'tone': 'rude'}  # empty name, invalid tone
    result = greeting_router.execute(invalid_input)
    
    print(f"Success: {result.success}")
    print("Errors:")
    for error in result.errors:
        print(f"  • {error}")
    print()


def demo_code_review_prompt():
    """Demo 2: Code review with structured output validation"""
    print("🔍 CODE REVIEW PROMPT DEMO")
    print("=" * 50)
    
    code_review_router = create_router(
        inputs={
            'code': {'type': 'string', 'min_length': 10, 'required': True},
            'language': {'type': 'string', 'enum': ['python', 'javascript', 'java', 'go'], 'required': True},
            'review_level': {'type': 'string', 'enum': ['beginner', 'intermediate', 'advanced'], 'required': True}
        },
        outputs={
            'overall_score': {'type': 'integer', 'min': 1, 'max': 10},
            'issues_found': {'type': 'array'},
            'suggestions': {'type': 'string', 'min_length': 1},
            'approved': {'type': 'boolean'}
        },
        prompt="""
Review this {{ language }} code for a {{ review_level }} developer:

```{{ language }}
{{ code }}
```

Provide:
1. Overall score (1-10)
2. List of issues found
3. Improvement suggestions
4. Approval recommendation

Be constructive and educational.
"""
    )
    
    # Test with sample code
    sample_input = {
        'code': 'def hello():\n    print("hello world")\n    return',
        'language': 'python',
        'review_level': 'beginner'
    }
    
    result = code_review_router.execute(sample_input)
    
    print(f"Input Validation: {'✅ PASSED' if result.input_validation['valid'] else '❌ FAILED'}")
    print(f"Rendered Prompt Length: {len(result.prompt_used)} characters")
    print("\nPrompt Preview:")
    print(result.prompt_used[:200] + "..." if len(result.prompt_used) > 200 else result.prompt_used)
    print()


def demo_content_generation():
    """Demo 3: Blog post generation with strict requirements"""
    print("📝 CONTENT GENERATION DEMO")
    print("=" * 50)
    
    blog_router = create_router(
        inputs={
            'topic': {'type': 'string', 'min_length': 5, 'required': True},
            'target_audience': {'type': 'string', 'enum': ['beginners', 'professionals', 'experts'], 'required': True},
            'word_count': {'type': 'integer', 'min': 100, 'max': 2000, 'required': True},
            'include_code': {'type': 'boolean', 'required': False}
        },
        outputs={
            'title': {'type': 'string', 'min_length': 10, 'max_length': 100},
            'content': {'type': 'string', 'min_length': 50},
            'tags': {'type': 'array'},
            'reading_time': {'type': 'integer', 'min': 1}
        },
        prompt="""
Write a blog post about "{{ topic }}" for {{ target_audience }}.

Requirements:
- Target length: {{ word_count }} words
- Audience: {{ target_audience }}
{% if include_code %}
- Include relevant code examples
{% endif %}

Create:
1. Engaging title (10-100 chars)
2. Well-structured content
3. Relevant tags (array)
4. Estimated reading time (minutes)

Make it informative and engaging.
"""
    )
    
    # Test multiple scenarios
    test_cases = [
        {
            'topic': 'Python decorators',
            'target_audience': 'beginners',
            'word_count': 500,
            'include_code': True
        },
        {
            'topic': 'Machine Learning Ethics',
            'target_audience': 'professionals',
            'word_count': 1200,
            'include_code': False
        },
        # Invalid case
        {
            'topic': 'AI',  # too short
            'target_audience': 'everyone',  # invalid enum
            'word_count': 50,  # too short
        }
    ]
    
    print("Running batch tests...")
    results = blog_router.test_prompt(test_cases)
    
    for i, result in enumerate(results):
        print(f"\n📊 Test Case {i+1}: {'✅ PASSED' if result.success else '❌ FAILED'}")
        if not result.success:
            print("Errors:")
            for error in result.errors:
                print(f"  • {error}")
        else:
            print(f"Execution time: {result.execution_time:.3f}s")
    print()


def demo_validation_only():
    """Demo 4: Using CALYX for input validation without LLM calls"""
    print("🛡️ VALIDATION-ONLY DEMO")
    print("=" * 50)
    
    api_validator = create_router(
        inputs={
            'user_id': {'type': 'integer', 'min': 1, 'required': True},
            'email': {'type': 'string', 'min_length': 5, 'required': True},
            'age': {'type': 'integer', 'min': 13, 'max': 120, 'required': True},
            'preferences': {'type': 'array', 'required': False}
        },
        outputs={},  # No outputs needed for validation-only
        prompt=""    # No prompt needed
    )
    
    # Test various inputs
    test_inputs = [
        {'user_id': 123, 'email': 'user@example.com', 'age': 25},  # valid
        {'user_id': -1, 'email': 'bad', 'age': 200},  # all invalid
        {'email': 'user@example.com', 'age': 25},  # missing required
    ]
    
    for i, test_input in enumerate(test_inputs):
        result = api_validator.validate_inputs(test_input)
        print(f"Input {i+1}: {'✅ VALID' if result.success else '❌ INVALID'}")
        if not result.success:
            for error in result.errors:
                print(f"  • {error}")
    print()


def demo_audit_trail():
    """Demo 5: Show the audit trail feature"""
    print("📋 AUDIT TRAIL DEMO")
    print("=" * 50)
    
    simple_router = create_router(
        inputs={'message': {'type': 'string', 'required': True}},
        outputs={'response': {'type': 'string'}},
        prompt="Echo: {{ message }}"
    )
    
    result = simple_router.execute({'message': 'Hello CALYX!'})
    
    print("Audit Trail:")
    for step in result.audit_trail:
        print(f"  🔍 {step}")
    
    print(f"\nExecution Summary:")
    print(f"  Success: {result.success}")
    print(f"  Time: {result.execution_time:.3f}s")
    print(f"  Input Valid: {result.input_validation['valid']}")
    print(f"  Output Valid: {result.output_validation['valid']}")
    print()


if __name__ == "__main__":
    print("🚀 CALYX: Statically Typed Prompts for Python")
    print("=" * 60)
    print("Write prompts like functions. Enforce structure like Pydantic.\n")
    
    demo_greeting_prompt()
    demo_code_review_prompt()
    demo_content_generation()
    demo_validation_only()
    demo_audit_trail()
    
    print("=" * 60)
    print("🎉 CALYX: Finally, prompt engineering you can debug, test, and trust!")
    print("⭐ No more prompt soup. No more JSON schema duct tape.")
    print("🔧 Build cognitive software like real software.")
