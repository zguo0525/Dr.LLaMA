import openai
import time
import numpy as np
from tqdm import tqdm

# models
GPT_MODEL = "gpt-3.5-turbo"

returned_data = []

for i in range(0, 50):
    for j in tqdm(range(10)):
        
        idx = i * 10 + j

        # an example question in PubMedQA
        query = inputs[idx]
        print(query)

        response = openai.ChatCompletion.create(
            messages=[
                {'role': 'system', 'content': 'answer the question given the context using a single word from below: yes, no, maybe'},
                {'role': 'user', 'content': query},
            ],
            model=GPT_MODEL,
            max_tokens=512,
            n=1,
            temperature=0.8,
#             repetition_penalty=2.0,
#             beam=5,
        )

        returned_data.append(response['choices'][0]['message']['content'])
        np.save('gpt3.5_answer_' + str(i) + '.npy', returned_data)
        #print(idx, returned_data[idx])
        time.sleep(1)
        #print(response['choices'][0]['message']['content'])
