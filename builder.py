from openai import OpenAI

client = OpenAI()

def generate_pipeline_code(task_description: str) -> str:
    """Ask the LLM to write LangChain code for the given task."""
    prompt = f'''
    You are an expert ML engineer. 
    Write *pure* Python code using LangChain that performs this task:

    {task_description}

    Requirements:
    - Load a text file named 'data/about_python.txt'.
    - Use FAISS and OpenAIEmbeddings for retrieval.
    - Use ChatOpenAI for the LLM.
    - Answer a user question and print only the final answer.

    Return only executable Python code.
    Do NOT include markdown fences (no ``` or ```python).
    Do NOT include explanations or comments before or after the code.
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
