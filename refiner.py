from openai import OpenAI
client = OpenAI()

def refine_pipeline(old_code: str, feedback: str) -> str:
    """Use the LLM to improve a low-performing pipeline."""
    prompt = f'''
    The following LangChain pipeline underperformed or crashed.
    Feedback or error details:
    {feedback}

    Please improve or fix the code while keeping the same purpose and structure.
    Use the same import pattern:
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import RetrievalQA

    Do NOT include markdown fences (no ``` or ```python).
    Do NOT include explanations or comments before or after the code.
    
    CODE:
    {old_code}
    '''
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}]
    )
    code = response.choices[0].message.content.strip()
    return code.replace('```python', '').replace('```', '').strip()