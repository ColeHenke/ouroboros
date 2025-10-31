import subprocess
from openai import OpenAI
import json

client = OpenAI()


def sanitize_code(code: str) -> str:
    """Sanitize code by removing Python markdown fences incase prompt directives are not followed."""
    return code.replace('```python', '').replace('```', '').strip()


def run_pipeline(code: str) -> tuple[str, str]:
    """Run generated code safely in a subprocess and capture stdout and/or stderr."""
    result = subprocess.run(
        ['python3', '-c', code],
        capture_output=True, text=True, timeout=20
    )

    return result.stdout.strip(), result.stderr.strip()


def validate_imports():
    required_modules = ['langchain', 'langchain_openai', 'langchain_community']
    for mod in required_modules:
        try:
            __import__(mod)
        except ImportError as e:
            print(f'Missing or invalid import: {mod}', e)


def evaluate_pipeline(code: str, question='Evaluate how true these statements are') -> tuple[float, str, str]:
    """Execute pipeline and grade factual correctness."""
    output, error = run_pipeline(code)

    if error:
        return 0.0, output, error

    judge_prompt = f'''
    Question: {question}
    Answer: {output}
    Score factual correctness on a scale from 0 to 1.
    Only return the numeric score.
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': judge_prompt}]
    )
    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text.split()[0])
    except ValueError:
        score = 0.0
    return score, output, error
