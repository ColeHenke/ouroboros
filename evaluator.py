# evaluator.py
import subprocess
from openai import OpenAI

client = OpenAI()

def run_pipeline(code: str) -> str:
    """Run generated code safely in a subprocess and capture stdout."""
    result = subprocess.run(
        ['python3', '-c', code],
        capture_output=True, text=True, timeout=20
    )
    return result.stdout.strip() or result.stderr.strip()

def evaluate_pipeline(code: str, question='Who created Python?') -> tuple[float, str]:
    """Execute pipeline and grade factual correctness."""
    output = run_pipeline(code)

    judge_prompt = f'''
    Question: {question}
    Answer: {output}
    Score factual correctness on a scale from 0 to 1.
    Only return the numeric score.
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": judge_prompt}]
    )
    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text.split()[0])
    except ValueError:
        score = 0.0
    return score, output
