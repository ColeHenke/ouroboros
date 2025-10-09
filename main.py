from dotenv import load_dotenv
load_dotenv()

import os, json
from builder import generate_pipeline_code
from evaluator import evaluate_pipeline
from refiner import refine_pipeline

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    with open('data/about_python.txt', 'w') as f:
        f.write('Python was created by Guido van Rossum in the late 1980s.')

    task = 'Build a LangChain pipeline that answers factual questions about the text file.'
    code = generate_pipeline_code(task)

    for i in range(3):
        print(f'\n=== Iteration {i+1} ===')
        score, output = evaluate_pipeline(code)
        print('Output:', output)
        print('Score:', score)

        with open(f'logs/generated_code_iter{i}.py', 'w') as f:
            f.write(code)

        with open('logs/results.jsonl', 'a') as log:
            json.dump({'iteration': i+1, 'score': score, 'output': output}, log)
            log.write('\n')

        if score >= 0.8:
            print('Acceptable performance reached.')
            break
        else:
            code = refine_pipeline(code, f'Score {score}, answer not correct enough.')
            print('Refining pipeline...')

if __name__ == '__main__':
    main()
