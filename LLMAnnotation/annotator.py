import os

import ujson as json
from func_timeout import func_set_timeout
from openai.error import RateLimitError
from sentence_transformers import SentenceTransformer
from ujson import JSONDecodeError

from LLMAnnotation.connector import Connector


class Annotator(Connector):
    def __init__(self, engine: str = 'gpt-3.5-turbo', config_path: str = None, 
                 description: str = None, guidance: str = None):
       
        super().__init__(engine)
        if config_path is None:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.join(dir_path, 'configs', 'fake_news_detection.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file missing: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.description = config['description'] if description is None else description
        self.guidance = config['guidance'] if guidance is None else guidance

    def __get_prompt(self, sample, demo=None):
        """
        Constructs a prompt for fake news detection.
        """
        demo_string = ""
        if demo is not None:
            for example in demo:
                demo_string += (
                    f"Example:\n{example['text']}\nLabel: {example['label']}\n\n"
                )

        task_string = (
            f"Analyze the following text and determine if it is Fake or Real:\n"
            f"{sample['text']}\nLabel:"
        )

        return '\n'.join([self.description, self.guidance, demo_string, task_string])

    def __postprocess(self, result):
        result = result.strip().lower()
        if result in ['fake', 'real']:
            return result.capitalize()
        return None

    @func_set_timeout(60)
    def online_annotate(self, sample, demo=None, return_cost: bool = False):
        """
        Annotates a single sample.
        """
        prompt = self.__get_prompt(sample, demo)
        try:
            response = self.online_query(prompt)
        except RateLimitError:
            raise RateLimitError()
        except Exception:
            if return_cost:
                return None, 0
            return None

        result = response['result']
        processed_result = self.__postprocess(result)

        if return_cost:
            return processed_result, response['cost']
        return processed_result

    @func_set_timeout(30)
    def batch_annotate(self, inputs, return_cost: bool = False):
        """
        Annotates multiple samples in a batch.
        Each input is a dictionary with {'sample': sample, 'demo': demo}.
        """
        if self.engine == 'gpt4':
            raise ValueError('ChatCompletion API does not support batch inference.')

        prompts = [self.__get_prompt(x['sample'], x['demo']) for x in inputs]
        try:
            response = self.batch_query(prompts)
        except Exception:
            print(f"Exception occurs with inputs: {prompts}")
            if return_cost:
                return [None for _ in range(len(inputs))], 0
            return [None for _ in range(len(inputs))]

        results = []
        for res in response['results']:
            processed_result = self.__postprocess(res)
            results.append(processed_result)

        if return_cost:
            return results, response['cost']
        return results


if __name__ == '__main__':
    
    annotator = Annotator(engine='gpt-3.5-turbo', config_name='fake_news_detection')
    sample = {"text": "The government has announced free healthcare for all citizens starting next year."}
    demo = [
        {"text": "Vaccines cause autism according to recent studies.", "label": "Fake"},
        {"text": "The Earth revolves around the Sun.", "label": "Real"}
    ]
    print(annotator.online_annotate(sample, demo))
