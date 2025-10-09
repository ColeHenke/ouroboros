# refiner.py
from openai import OpenAI
client = OpenAI()

def refine_pipeline(old_code: str, feedback: str) -> str:
    """Use the LLM to improve a low-performing pipeline."""
    prompt = f'''
    The following LangChain pipeline underperformed.
    Feedback: {feedback}

    Please improve it while keeping the same purpose and structure.
    Return only executable Python code.
    Do NOT include markdown fences (no ``` or ```python).
    Do NOT include explanations or comments before or after the code.
    
    CODE:
    {old_code}
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
