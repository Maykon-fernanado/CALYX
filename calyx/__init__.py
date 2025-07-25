"""
CALYX: Statically Typed Prompts for Python
🔧 Write prompts like functions. Enforce structure like Pydantic.

CALYX (YAML + JSON + Prompt + Human) is a type-safe prompt engineering layer.
"""

import yaml
import json
import re
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime


class ValidationError(Exception):
    """Raised when input/output validation fails"""
    def __init__(self, message: str, field: str = None, expected: Any = None, actual: Any = None):
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(message)


@dataclass
class PromptResult:
    """The result of a prompt execution with full traceability"""
    success: bool
    output: Optional[Dict[str, Any]] = None
    errors: List[str] = None
    prompt_used: Optional[str] = None
    input_validation: Dict[str, Any] = None
    output_validation: Dict[str, Any] = None
    execution_time: Optional[float] = None
    model_used: Optional[str] = None
    audit_trail: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.audit_trail is None:
            self.audit_trail = []


class TypeValidator:
    """Validates data against YAML schema definitions"""
    
    @staticmethod
    def validate_field(value: Any, schema: Dict[str, Any], field_name: str) -> tuple[bool, str]:
        """Validate a single field against its schema"""
        
        # Check type
        expected_type = schema.get('type', 'string')
        
        if expected_type == 'string':
            if not isinstance(value, str):
                return False, f"{field_name}: expected string, got {type(value).__name__}"
            
            # Check min_length
            if 'min_length' in schema and len(value) < schema['min_length']:
                return False, f"{field_name}: minimum length {schema['min_length']}, got {len(value)}"
            
            # Check max_length
            if 'max_length' in schema and len(value) > schema['max_length']:
                return False, f"{field_name}: maximum length {schema['max_length']}, got {len(value)}"
            
            # Check enum
            if 'enum' in schema and value not in schema['enum']:
                return False, f"{field_name}: must be one of {schema['enum']}, got '{value}'"
                
        elif expected_type == 'integer':
            if not isinstance(value, int):
                return False, f"{field_name}: expected integer, got {type(value).__name__}"
            
            # Check min/max
            if 'min' in schema and value < schema['min']:
                return False, f"{field_name}: minimum value {schema['min']}, got {value}"
            if 'max' in schema and value > schema['max']:
                return False, f"{field_name}: maximum value {schema['max']}, got {value}"
                
        elif expected_type == 'number':
            if not isinstance(value, (int, float)):
                return False, f"{field_name}: expected number, got {type(value).__name__}"
                
        elif expected_type == 'boolean':
            if not isinstance(value, bool):
                return False, f"{field_name}: expected boolean, got {type(value).__name__}"
                
        elif expected_type == 'array':
            if not isinstance(value, list):
                return False, f"{field_name}: expected array, got {type(value).__name__}"
        
        return True, ""
    
    @staticmethod
    def validate_data(data: Dict[str, Any], schema: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate entire data structure against schema"""
        errors = []
        
        # Check required fields
        required_fields = [field for field, config in schema.items() 
                          if isinstance(config, dict) and config.get('required', False)]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Required field '{field}' is missing")
        
        # Validate each field present in data
        for field_name, value in data.items():
            if field_name in schema:
                field_schema = schema[field_name]
                if isinstance(field_schema, dict):
                    is_valid, error_msg = TypeValidator.validate_field(value, field_schema, field_name)
                    if not is_valid:
                        errors.append(error_msg)
        
        return len(errors) == 0, errors


class PromptTemplate:
    """Handles prompt template rendering with Jinja2-like syntax"""
    
    @staticmethod
    def render(template: str, variables: Dict[str, Any]) -> str:
        """Render template with variables using simple {{ }} syntax"""
        rendered = template
        
        # Simple variable substitution for {{ variable }}
        for key, value in variables.items():
            pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
            rendered = re.sub(pattern, str(value), rendered)
        
        return rendered


class Router:
    """The core CALYX prompt routing engine"""
    
    def __init__(self, schema_file: str = None, schema_dict: Dict[str, Any] = None):
        """Initialize with either a YAML file or dictionary schema"""
        self.schema = {}
        self.llm_client = None
        
        if schema_file:
            with open(schema_file, 'r') as f:
                self.schema = yaml.safe_load(f)
        elif schema_dict:
            self.schema = schema_dict
        
        # Extract components
        self.inputs_schema = self.schema.get('inputs', {})
        self.outputs_schema = self.schema.get('outputs', {})
        self.prompt_template = self.schema.get('prompt', '')
        self.model_config = self.schema.get('model', {})
    
    def set_llm_client(self, client: Any):
        """Set the LLM client (OpenAI, Anthropic, etc.)"""
        self.llm_client = client
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> PromptResult:
        """Validate inputs against schema"""
        start_time = datetime.now()
        
        is_valid, errors = TypeValidator.validate_data(inputs, self.inputs_schema)
        
        if not is_valid:
            return PromptResult(
                success=False,
                errors=errors,
                input_validation={'valid': False, 'errors': errors},
                execution_time=(datetime.now() - start_time).total_seconds()
            )
        
        return PromptResult(
            success=True,
            input_validation={'valid': True, 'errors': []},
            execution_time=(datetime.now() - start_time).total_seconds()
        )
    
    def render_prompt(self, inputs: Dict[str, Any]) -> str:
        """Render the prompt template with validated inputs"""
        return PromptTemplate.render(self.prompt_template, inputs)
    
    def validate_outputs(self, outputs: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate LLM outputs against schema"""
        if not self.outputs_schema:
            return True, []
        
        return TypeValidator.validate_data(outputs, self.outputs_schema)
    
    def execute(self, inputs: Dict[str, Any], llm_client: Any = None) -> PromptResult:
        """Execute the full CALYX pipeline: validate -> render -> call LLM -> validate output"""
        start_time = datetime.now()
        audit_trail = ['pipeline_started']
        
        # Step 1: Validate inputs
        audit_trail.append('validating_inputs')
        input_validation = self.validate_inputs(inputs)
        if not input_validation.success:
            input_validation.audit_trail = audit_trail + ['input_validation_failed']
            return input_validation
        
        # Step 2: Render prompt
        audit_trail.append('rendering_prompt')
        try:
            rendered_prompt = self.render_prompt(inputs)
        except Exception as e:
            return PromptResult(
                success=False,
                errors=[f"Prompt rendering failed: {str(e)}"],
                audit_trail=audit_trail + ['prompt_rendering_failed']
            )
        
        # Step 3: Call LLM (if client provided)
        audit_trail.append('calling_llm')
        client = llm_client or self.llm_client
        llm_response = None
        
        if client:
            try:
                # This would be implemented based on the specific LLM client
                # For now, return a mock response
                audit_trail.append('llm_call_successful')
                llm_response = {"status": "mock_response"}
            except Exception as e:
                return PromptResult(
                    success=False,
                    errors=[f"LLM call failed: {str(e)}"],
                    prompt_used=rendered_prompt,
                    audit_trail=audit_trail + ['llm_call_failed']
                )
        
        # Step 4: Validate outputs (if we have them)
        output_validation = {'valid': True, 'errors': []}
        if llm_response and isinstance(llm_response, dict):
            is_valid, errors = self.validate_outputs(llm_response)
            output_validation = {'valid': is_valid, 'errors': errors}
            
            if not is_valid:
                return PromptResult(
                    success=False,
                    errors=errors,
                    output=llm_response,
                    prompt_used=rendered_prompt,
                    input_validation=input_validation.input_validation,
                    output_validation=output_validation,
                    audit_trail=audit_trail + ['output_validation_failed']
                )
        
        audit_trail.append('pipeline_completed')
        
        return PromptResult(
            success=True,
            output=llm_response,
            prompt_used=rendered_prompt,
            input_validation=input_validation.input_validation,
            output_validation=output_validation,
            execution_time=(datetime.now() - start_time).total_seconds(),
            audit_trail=audit_trail
        )
    
    def test_prompt(self, test_cases: List[Dict[str, Any]]) -> List[PromptResult]:
        """Test the prompt with multiple input cases"""
        results = []
        for i, test_case in enumerate(test_cases):
            print(f"Running test case {i+1}/{len(test_cases)}")
            result = self.execute(test_case)
            results.append(result)
        return results


# Convenience functions for quick setup
def load_schema(file_path: str) -> Router:
    """Quick loader for YAML schema files"""
    return Router(schema_file=file_path)


def create_router(inputs: Dict[str, Any], outputs: Dict[str, Any], prompt: str) -> Router:
    """Create a router from inline definitions"""
    schema = {
        'inputs': inputs,
        'outputs': outputs,
        'prompt': prompt
    }
    return Router(schema_dict=schema)


# CLI entry point
def main():
    """CLI interface for CALYX"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CALYX: Statically Typed Prompts for Python")
    parser.add_argument('--schema', required=True, help='YAML schema file')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', help='Output JSON file (optional)')
    parser.add_argument('--validate-only', action='store_true', help='Only validate, don\'t call LLM')
    
    args = parser.parse_args()
    
    # Load schema and input
    router = load_schema(args.schema)
    
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    # Execute or validate
    if args.validate_only:
        result = router.validate_inputs(input_data)
        print("Input validation:", "PASSED" if result.success else "FAILED")
        if not result.success:
            for error in result.errors:
                print(f"  ❌ {error}")
    else:
        result = router.execute(input_data)
        
        # Output results
        result_dict = asdict(result)
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result_dict, f, indent=2)
        else:
            print(json.dumps(result_dict, indent=2))


if __name__ == "__main__":
    main()
