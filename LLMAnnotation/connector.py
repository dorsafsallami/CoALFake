import json
import os

import openai


class Connector:
    def __init__(self, engine='gpt-3.5-turbo'):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, 'openai_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.key = config['key']
        self.base = config['base']
        self.engine = engine
        
        openai.api_key = self.key
        openai.api_base = self.base

    def online_query(self, prompt):
        task_param = {
        'model': self.engine,
        'messages': [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        'temperature': 0,
        'max_tokens': 1000
        }

        response = openai.ChatCompletion.create(**task_param)
        result = response.choices[0].message['content'].strip()
        cost = response.usage['total_tokens']
    
        return {'result': result, 'cost': cost}
    
    def batch_query(self, prompts):
        results = []
        total_cost = 0

        for prompt in prompts:
            response = self.online_query(prompt)
            results.append(response['result'])
            total_cost += response['cost']

        print(f'The total cost of batch query is {total_cost} tokens.')
        return {'results': results, 'cost': total_cost}

if __name__ == '__main__':
    conn = Connector(engine='gpt-3.5-turbo') 
    prompts = [
        'How credible is the statement that vaccines cause autism?',
        'Analyze the reliability of the source claiming climate change is a hoax.',
        'Evaluate the truthfulness of the statement: "5G networks cause COVID-19."'
    ]
    responses = conn.batch_query(prompts)
    print(responses)
