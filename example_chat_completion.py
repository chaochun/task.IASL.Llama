# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
from typing import Optional

import fire

from llama import Llama


def demo_1(generator):
    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")

    return    

##@20230804
def demo_2(generator, max_gen_len, temperature, top_p):

##  loop query, w/o history
    while True:
        query = input('\nQuestion: ') 
        if not query:
            break

        prompt_user = "你要根據句子<>的內容，先找出所有句子中所描述的數字，\n這個數字稱為數詞。再找出對應的量詞與這個數字描述的名詞。\n\
    最後再寫出與這個數字相關的一個主要動詞與主要主詞(主詞須為名詞或代名詞）。\n如果沒有所需要的詞性的字，就以#表示。\n句子中的'每'，'一'，\"幾\"，與\"多少\" 都算是數詞。\n\
    寫出的答案用json格式回答(不要寫出與json無關的字串)，key為num，dm，noun，verb，subject。\n如果有多個json物件，則用list表示。\n\n例如: 如果輸入的句子是，\"小華買5個蘋果和小明賣3顆鳳梨\"，\
    你要回答:\n\n[\n{\n\"num\": \"5\",\n\"dm\": \"個\",\n\"noun\": \"蘋果\",\n\"verb\": \"買\",\n\"subject\": \"小華\"\n},\n\
    {\n\"num\": \"3\",\n\"dm\": \"顆\",\n\"noun\": \"鳳梨\",\n\"verb\": \"賣\",\n\"subject\": \"小明\"\n}\n]\n\n\n\
    例如: 如果輸入的句子是，\"每盒內是裝了幾顆水蜜桃\"，你要回答:\n\n[\n{\n\"num\": \"每\",\n\"dm\": \"盒\",\n\"noun\": \"水蜜桃\",\n\"verb\": \"裝\",\n\"subject\": \"#\"\n},\n\
    {\n\"num\": \"幾\",\n\"dm\": \"顆\",\n\"noun\": \"水蜜桃\",\n\"verb\": \"是\",\n\"subject\": \"#\"\n}\n]\n\n例如: 如果輸入的句子是，\"一盒水蜜桃有3顆\"，\
    你要回答:\n\n[\n{\n\"num\": \"一\",\n\"dm\": \"盒\",\n\"noun\": \"水蜜桃\",\n\"verb\": \"有\",\n\"subject\": \"#\"\n},\n\
    {\n\"num\": \"3\",\n\"dm\": \"顆\",\n\"noun\": \"水蜜桃\",\n\"verb\": \"有\",\n\"subject\": \"#\"\n}\n]\n\n例如: 如果輸入的句子是，\"1公斤香蕉賣幾元\"，\
    你要回答:\n\n[\n{\n\"num\": \"一\",\n\"dm\": \"公斤\",\n\"noun\": \"香蕉\",\n\"verb\": \"賣\",\n\"subject\": \"#\"\n},\n\
    {\n\"num\": \"幾\",\n\"dm\": \"元\",\n\"noun\": \"#\",\n\"verb\": \"賣\",\n\"subject\": \"#\"\n}\n]\n\n\
    例如: 如果輸入的句子是，\"我買了5+(1/2)盒水蜜桃\"，你要回答:\n\n[\n{\n\"num\": \"5+(1/2)\",\n\"dm\": \"盒\",\n\"noun\": \"水蜜桃\",\n\"verb\": \"買\",\n\"subject\": \"我\"\n}\n]\n\
    \n句子<{%s}>"%(query)

        dialogs = [[{"role": "user", "content": prompt_user}]]

        print('>>asking')
        results = generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        print('>>response')

        dialog = dialogs[-1]
        result = results[-1
        ]
        answer = result['generation']['content']
        answer = answer.strip()
        print('A:', answer)

    return
##@20230804

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048, ##@20230804, default=512
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):

    if torch.cuda.is_available():
        print('>>CUDA GPU is available')
    else:
        print('>>RUN in CPU mode')

    # generator = None
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

##  @20230804
    # demo_1(generator)
##  @20230804

##  @20230804
    demo_2(generator, max_gen_len, temperature, top_p)
##  @20230804

if __name__ == "__main__":
    fire.Fire(main)
