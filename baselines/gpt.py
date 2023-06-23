import json
import logging
import time
from timeit import default_timer as timer

import openai
import tiktoken
from tqdm import tqdm

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

logging.basicConfig(level=logging.INFO, filename=f'logs/gpt_prompt_{timestamp}.log', filemode='w')


class ChatApp:
    def __init__(self, r_code2instances, gpt_example_gen, uids_to_ignore, dataset, n_samples=2):
        # Setting the API key to use the OpenAI API
        # openai.api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = None
        self.n_samples = n_samples
        self.dataset = dataset

        # Get n known and n novel
        self.n_known, self.n_novel = 0, 0
        self.n_known_and_novel()

        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.r_code2instances = r_code2instances
        self.gpt_helper = gpt_example_gen
        self.uids_to_ignore = uids_to_ignore
        self.system_instructions = 'None'

        self.prompt_intructions()
        self.system_prompt = {
            "role": "system",
            "content": self.system_instructions
        }

    def n_known_and_novel(self):
        if self.dataset == 'fewrel':
            self.n_known = 40
            self.n_novel = 80
        elif self.dataset == 'tacred':
            self.n_known = 20
            self.n_novel = 21
        elif self.dataset == 'retacred':
            self.n_known = 19
            self.n_novel = 20
        else:
            raise NotImplementedError

    # Send it to the master (GPT-3) and get a response
    def chat(self, message, uid):
        user_prompt = {
            "role": "user",
            "content": message
        }
        messages = [self.system_prompt, user_prompt]
        try:
            start = timer()
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                max_tokens=15,
                messages=messages
            )
            end = timer()
            elapsed = end - start
            return response["choices"][0]["message"]["content"], elapsed
        except Exception as e:
            logging.error(f'Error in chat with uid: {uid}')
            logging.error(e)
            return f'Error: {e}'

    # Function to estimate the number of tokens in a message
    def token_count(self, message):
        return len(self.encoding.encode(message))

    # Generate instuctions:
    def prompt_intructions(self):
        known_relation_names = ', '.join([f'"{x}"' for x in self.gpt_helper.known_rel_names[1:]])  # Skip 'NA'
        examples = ''
        for i, (rel_code, instances) in enumerate(self.r_code2instances.items()):
            instance = instances[42]  # Randomly select one instance
            example, label = self.gpt_helper.raw_ins2ex_and_label(instance)
            if self.dataset == 'fewrel':
                label_name = self.gpt_helper.rel_code2name[label]
            else:
                label_name = label
            examples += f'\n- {example}\n"{label_name}"\n'
            if i == self.n_samples - 1:
                break

        instructions = \
            '1. Select the correct relation between the head and tail entities in the following unlabeled examples. ' \
            '\n2. Each example has the head and tail entities appended to the sentence in the form: (head entity) (tail entity). ' \
            f'\n3. There are {self.n_known} known relation classes, and up to {self.n_novel} unknown, or novel, relation classes. ' \
            '\n4. The following is the list of known relation classes: ' \
            f'{known_relation_names} ' \
            '\n5. If the instance is a novel class, suggest the most likely novel class name. ' \
            '\n6. Here are some examples:\n' \
            f'{examples}' \
            '\n7. Respond only with the class name in quotes:\n'

        logging.info(instructions)
        self.system_instructions = instructions

    def prompt_send_and_receive(self, dataloader, gpt_helper):
        with open(f'data/{self.dataset}/{self.dataset}_unlabeled_gpt_predictions.jsonl', 'a') as f:
            for i, batch in enumerate(tqdm(dataloader)):
                if self.dataset == 'fewrel':
                    rel_name = gpt_helper.rel_code2name[batch['label'][0]]
                else:
                    rel_name = batch['label'][0]
                instance = {'prompt': batch['instance'][0],
                            'label': batch['label'][0],
                            'uid': batch['uid'][0].item(),
                            'rel_name': rel_name,
                            'is_known': batch['label'][0] in gpt_helper.known_rel_codes,
                            'rel_id': gpt_helper.rel2id[batch['label'][0]],
                            'response': '',
                            }
                # Skip some UIDs
                if instance['uid'] in self.uids_to_ignore:
                    logging.info(f'Skipping uid: {instance["uid"]}')
                    continue

                # Send prompt
                response, time_elapsed = self.chat(instance['prompt'], instance['uid'])
                instance['response'] = response.replace('\"', '')

                # Log response
                logging.info(f'{instance}')
                f.write(f'{json.dumps(instance)}\n')

                # Ensure we don't send too many requests
                sleep_time = max(0, 0.36 - time_elapsed)  # max = 166 requests per minute
                time.sleep(sleep_time)
