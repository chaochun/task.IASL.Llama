# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
from typing import Optional

import fire

from llama import Llama


def demo_1(generator, max_gen_len, temperature, top_p):
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

def load_prompt(load_chinese_prompt=True):

    if load_chinese_prompt:
        # ifname = 'prompts_chn.txt' ## w/ instruction
        ifname = 'prompts_chn2.txt' ## w/o instruction
        ihandle = open(ifname, encoding='UTF-8')
    else:
        # ifname = 'prompts_eng.txt'
        ifname = 'prompts_eng__simple.txt'
        ihandle = open(ifname)

    lines = ihandle.readlines()
    outline = ''
    for line in lines:
        line = line.strip()
        outline += line

    return outline

##@20230804
def demo_2_interactive(generator, max_gen_len, temperature, top_p):

    # USE_CHN_PROMPT = True
    USE_CHN_PROMPT = False
    
    print('>>load_prompt ___begin')
    sys_prompt = load_prompt(load_chinese_prompt=USE_CHN_PROMPT)
    print('>>load_prompt ___end')

##  loop query, w/ history
    # dialogs = []
    while True:
##      loop query, w/o history
        dialogs = []

        query = input('\nQuestion: ') 
        if not query:
            break

        print('>>prompt question: %s'%(query))

        if USE_CHN_PROMPT:
            query = "%s  你要分析的句子是<S>%s</S>"%(sys_prompt, query)
        else:
            query = "%s  Analyze the sentence in <S>%s</S>"%(sys_prompt, query)

        print(query)

        dialog = [{"role": "user", "content": "%s"%(query)}]
        dialogs.append(dialog)
##  @20230807

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


##@20230810
def demo_2_given_queries(generator, max_gen_len, temperature, top_p):

    USE_CHN_PROMPT = True
    # USE_CHN_PROMPT = False

    sentences = ["小華買5個蘋果", \
    "小華買5個蘋果和小明賣3顆鳳梨", \
    "我買4瓶烏龍茶", \
    "我買了1/2盒水蜜桃", \
    "總共裝了幾顆水蜜桃", \
    "每盒裝三顆水蜜桃", \
    "一根長2.75公尺的鋼筋", \
    "阿智有7.32盒筆芯", \
    "阿智和言言共有7.32盒筆芯", \
    "已知阿智有4.62盒筆"]
    # sentences = ["小華買5個蘋果", \
    # "小華買5個蘋果和小明賣3顆鳳梨"]

    print('>>load_prompt ___begin')
    sys_prompt = load_prompt(load_chinese_prompt=USE_CHN_PROMPT)
    print('>>load_prompt ___end')

    for query in sentences:
##      loop query, w/o history
        dialogs = []

        print('>>prompt question: %s'%(query))

        if USE_CHN_PROMPT:
            query = "%s  你要分析的句子是<S>%s</S>"%(sys_prompt, query)
        else:
            query = "%s  Analyze the sentence in <S>%s</S>"%(sys_prompt, query)

        print(query)

        dialog = [{"role": "user", "content": "%s"%(query)}]
        dialogs.append(dialog)
##  @20230807

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

    print('886!')
    return

def get_MWP_CHN_demo_sentences__unitmap():
    sentences = ["小華買5個蘋果", \
    "小華買5個蘋果和小明賣3顆鳳梨", \
    "我買4瓶烏龍茶", \
    "我買了1/2盒水蜜桃", \
    "總共裝了幾顆水蜜桃", \
    "每盒裝三顆水蜜桃", \
    "一盒裝三顆水蜜桃", \
    "一根長2.75公尺的鋼筋", \
    "阿智有3.5盒筆芯", \
    "阿智和言言共有7.32盒筆芯", \
    "已知阿智有4.62盒筆", \
    "鋼筋重20公斤", \
    "相同的鋼筋1/2公尺長重多少公斤", \
    "學校的圍牆高2.1公尺", \
    "2.1公尺大約是多少台尺？", \
    "柳丁平分裝成7袋", \
    "一袋柳丁是幾公斤", \
    "3瓶烏龍茶剛好可以倒滿7杯"]
    return sentences

def get_MWP_CHN_demo_sentences__sum():
    sentences = ["有一條繩子，", \
    "每2/9公尺剪一段，", \
    "2/9公尺剪一段，", \
    "可以剪成5段，", \
    "還剩下0.5公尺，", \
    "剩下0.5公尺，", \
    "這條繩子有多少公尺？", \
    "繩子有多少公尺？", \
    "一盒餅乾連盒子共重1050公克，", \
    "其中空盒子重125公克，", \
    "空盒子重125公克，", \
    "姨媽買了46盒餅乾，", 
    "其中的餅乾共重多少公克？", \
    "餅乾共重多少公克？"]

    return sentences


def demo_2A_given_queries(generator, max_gen_len, temperature, top_p):

    USE_CHN_PROMPT = True
    # USE_CHN_PROMPT = False

    if USE_CHN_PROMPT:
        sentences = get_MWP_CHN_demo_sentences__unitmap()
        # sentences = get_MWP_CHN_demo_sentences__sum()
    else:
        sentences = ['John bought an apple.', \
        'You walks to school.', \
        "I will see a move today"]

    print('>>load_prompt ___begin')
    sys_prompt = load_prompt(load_chinese_prompt=USE_CHN_PROMPT)
    print('>>load_prompt ___end')

    for query in sentences:
##      loop query, w/o history
        dialogs = []

        print('\n')
        print('>>prompt question: %s'%(query))

        if USE_CHN_PROMPT:
            query = "%s  你要分析的句子是<S>%s</S>"%(sys_prompt, query)
        else:
            query = "%s  Analyze the sentence in <S>%s</S>"%(sys_prompt, query)

        print(query)

        dialog = [{"role": "user", "content": "%s"%(query)}]
        dialogs.append(dialog)
##  @20230807

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

    print('886!')
    return    
##@20230810

##@20230810
def demo_zero(generator, max_gen_len, temperature, top_p):

##  loop query, w/ history
    dialogs = []
    while True:
##      loop query, w/o history
        dialogs = []

        query = input('\nQuestion: ') 
        if not query:
            break

        dialog = [{"role": "user", "content": "%s"%(query)}]
        dialogs.append(dialog)
##  @20230807

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
    temperature: float = 0.0, ##@20230804, default=0.6
    top_p: float = 0.9,
    max_seq_len: int = 2048, ##@20230804, default=512
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):

    load_prompt()

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
    # demo_1(generator, max_gen_len, temperature, top_p)
##  @20230804

##  @20230804
    # demo_2_given_queries(generator, max_gen_len, temperature, top_p)
    demo_2A_given_queries(generator, max_gen_len, temperature, top_p)
##  @20230804

    # demo_zero(generator, max_gen_len, temperature, top_p)

if __name__ == "__main__":
    fire.Fire(main)
