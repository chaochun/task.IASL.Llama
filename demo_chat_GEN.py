# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch
from typing import Optional

import fire

from llama import Llama



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

def get_MWP_CHN_demo_sentences__rephrasing():
    sentences = ["舉出10種跟肚子痛類似的說法"]
    return sentences

def get_MWP_CHN_demo_sentences__generating_sentences():
    sentences = ["造10個類似以下說法的句子。\n因為長時間注視電腦螢幕，我的眼睛乾澀難耐。"]
    return sentences

def get_MWP_CHN_demo_sentences__generating_dialogs():
    sentences = ["寫出10句醫生與年紀大(超過60歲)的病人之間可能的對話, 病人問問題 醫生回答。", \
    "寫出10句醫生與年紀小(不到10歲)的病人之間可能的對話, 病人問問題 醫生回答"]
    return sentences

##@20230822
def get_MWP_ENG_demo_sentences__solving_MWPs():
    case1 = "Solve the following math word problem step by step.\n\n\
    Seven red apples and two green apples are in the basket. \n\
    How many apples are in the basket?"

    case12 = "Solve the following math word problem.\n\n\
    Seven red apples and two green apples are in the basket. \n\
    How many apples are in the basket?"    

    case2 = "Solve the following math word problem step by step.\n\n\
    17 plums were in the basket. More plums were added to the basket. Now there are 21 plums. \n\
    How many plums were added to the basket?"

    case22 = "Solve the following math word problem.\n\n\
    17 plums were in the basket. More plums were added to the basket. Now there are 21 plums. \n\
    How many plums were added to the basket?"

    case3 = "Solve the following math word problem step by step.\n\n\
    Kevin collected toys to use as prizes at the fair. He collected 14 stuffed animals. \n\
    He also collected 18 frisbees and several yo-yos. Kevin has 50 prizes in all. \n\
    How many yo-yos did Kevin collect?"

    case32 = "Solve the following math word problem.\n\n\
    Kevin collected toys to use as prizes at the fair. He collected 14 stuffed animals. \n\
    He also collected 18 frisbees and several yo-yos. Kevin has 50 prizes in all. \n\
    How many yo-yos did Kevin collect?"

    case51 = "Solve the following math word problem.\n\n\
    My older brother bought a cake for 55 dollars and a bread, paying a total of 82 dollars. \n\
    How much does a bread cost?"

    case52 = "Solve the following math word problem.\n\n\
    In Class 1 of Grade 3, there are 18 boys and 12 girls. If the entire class is divided into 6 groups, \n\
    how many students are in each group?"

    case53 = "Solve the following math word problem.\n\n\
    Dad works 7 hours a day. \n\
    How many hours does he work in 5 days?"

    case54 = "Solve the following math word problem.\n\n\
    With 20 dollars you can buy 4 chocolates. \n\
    How many chocolates can you buy with 50 dollars?"

    case55 = "Solve the following math word problem.\n\n\
    A box of grapes including the box weighs 875 grams, with the empty box weighing 28 grams. \n\
    How much do 17 boxes of grapes weigh?"

    case56 = "Solve the following math word problem.\n\n\
    The teacher evenly divided 200 pencils among 6 groups of boys and 4 groups of girls. \n\
    Each group keeps 5 pencils. The remaining ones are bundled with rubber bands, 3 pencils per bundle. \n\
    How many bundles can each group make?"

    case61 = "Solve the following math word problem step by step.\n\n\
    My older brother bought a cake for 55 dollars and a bread, paying a total of 82 dollars. \n\
    How much does a bread cost?"

    case62 = "Solve the following math word problem step by step.\n\n\
    In Class 1 of Grade 3, there are 18 boys and 12 girls. If the entire class is divided into 6 groups, \n\
    how many students are in each group?"

    case63 = "Solve the following math word problem step by step.\n\n\
    Dad works 7 hours a day. \n\
    How many hours does he work in 5 days?"

    case64 = "Solve the following math word problem step by step.\n\n\
    With 20 dollars you can buy 4 chocolates. \n\
    How many chocolates can you buy with 50 dollars?"

    case65 = "Solve the following math word problem step by step.\n\n\
    A box of grapes including the box weighs 875 grams, with the empty box weighing 28 grams. \n\
    How much do 17 boxes of grapes weigh?"

    case66 = "Solve the following math word problem step by step.\n\n\
    The teacher evenly divided 200 pencils among 6 groups of boys and 4 groups of girls. \n\
    Each group keeps 5 pencils. The remaining ones are bundled with rubber bands, 3 pencils per bundle. \n\
    How many bundles can each group make?"

    sentences = [case1, case12, case2, case22, case3, case32, \
    case61, case51, \
    case62, case52, \
    case63, case53, \
    case64, case54, \
    case65, case55, \
    case66, case56]

    return sentences

def get_MWP_CHN_demo_sentences__solving_MWPs():
    sentences = ["哥哥買一塊55 元的蛋糕和一個麵包，共付了 82 元，一個麵包要多少元？", \
    "三年1班有男生18人、女生12人，全班打散分成6組，每組有多少人？", \
    "爸爸一天工作7小時，5天工作幾小時？", \
    "20元可以買4顆巧克力，50元可以買幾顆巧克力？", \
    "一盒葡萄連盒子共重875公克，其中空盒子重28公克，17盒的葡萄重多少公克？",\
    "老師把200枝筆平分給6組男生和4組女生，每組留下5枝，剩下的再用橡皮筋將每3枝捆成一份，每組可以捆成幾份？", \
    "將下面的數學問題先翻譯成英文再用英文回答。\n三年1班有男生18人、女生12人，全班打散分成6組，每組有多少人？"]
    return sentences

# def get_MWP_CHN_demo_sentences__solving_MWPs__2(): ##@20230821, 比照::get_MWP_CHN_demo_sentences__summarization(), 先把"繁體中文"提示寫在最前面 -> 沒幫助
#     sentences = ["請用繁體中文回答以下數學問題\n\n哥哥買一塊55 元的蛋糕和一個麵包，共付了 82 元，一個麵包要多少元？", \
#     "請用繁體中文回答以下數學問題\n\n三年1班有男生18人、女生12人，全班打散分成6組，每組有多少人？", \
#     "請用繁體中文回答以下數學問題\n\n爸爸一天工作7小時，5天工作幾小時？", \
#     "請用繁體中文回答以下數學問題\n\n20元可以買4顆巧克力，50元可以買幾顆巧克力？", \
#     "請用繁體中文回答以下數學問題\n\n一盒葡萄連盒子共重875公克，其中空盒子重28公克，17盒的葡萄重多少公克？",\
#     "請用繁體中文回答以下數學問題\n\n老師把200枝筆平分給6組男生和4組女生，每組留下5枝，剩下的再用橡皮筋將每3枝捆成一份，每組可以捆成幾份？", \
#     "請用繁體中文回答以下數學問題\n\n將下面的數學問題先翻譯成英文再用英文回答。\n三年1班有男生18人、女生12人，全班打散分成6組，每組有多少人？"]
#     return sentences

def get_MWP_CHN_demo_sentences__summarization():
    case1 = "請用20個字摘要以下的新聞\n\n\
    中央氣象局今天表示，第6號颱風卡努已達中颱上限，目前中心在台北東北東方海面，向西北西移動，對台灣北部海面及東北部海面將構成威脅，持續發布海上颱風警報。\
    卡努今天降雨區主要在北台灣，因卡努行進方向偏西，暴風圈有可能接觸北台灣陸地，預估明後天（3、4日）最接近台灣，若路徑沒有變化，預計今天下午或傍晚將發布陸上颱風警報，\
    受影響範圍包括基隆北海岸、東北角。"

    case2 = "請用20個字摘要以下的新聞(以繁體中文回答)\n\n\
    中央氣象局今天表示，第6號颱風卡努已達中颱上限，目前中心在台北東北東方海面，向西北西移動，對台灣北部海面及東北部海面將構成威脅，持續發布海上颱風警報。\
    卡努今天降雨區主要在北台灣，因卡努行進方向偏西，暴風圈有可能接觸北台灣陸地，預估明後天（3、4日）最接近台灣，若路徑沒有變化，預計今天下午或傍晚將發布陸上颱風警報，\
    受影響範圍包括基隆北海岸、東北角。"

    case3 = "請用20個繁體中文字摘要以下的新聞\n\n\
    中央氣象局今天表示，第6號颱風卡努已達中颱上限，目前中心在台北東北東方海面，向西北西移動，對台灣北部海面及東北部海面將構成威脅，持續發布海上颱風警報。\
    卡努今天降雨區主要在北台灣，因卡努行進方向偏西，暴風圈有可能接觸北台灣陸地，預估明後天（3、4日）最接近台灣，若路徑沒有變化，預計今天下午或傍晚將發布陸上颱風警報，\
    受影響範圍包括基隆北海岸、東北角。"

    case4 = "請用繁體中文字並針對以下四項摘要新聞的重點\n\
    1. 颱風編號\n\
    2. 颱風中心位置\n\
    3. 影響區域\n\
    4. 預估發布警報時間\n\n\
    中央氣象局今天表示，第6號颱風卡努已達中颱上限，目前中心在台北東北東方海面，向西北西移動，對台灣北部海面及東北部海面將構成威脅，持續發布海上颱風警報。\
    卡努今天降雨區主要在北台灣，因卡努行進方向偏西，暴風圈有可能接觸北台灣陸地，預估明後天（3、4日）最接近台灣，若路徑沒有變化，預計今天下午或傍晚將發布陸上颱風警報，\
    受影響範圍包括基隆北海岸、東北角。"

    sentences = [case1, case2, case3, case4]
    return sentences

def get_MWP_CHN_demo_sentences__KGGeneration():
    case1 = "請用繁體中文字產生\"旅行\"的事理圖譜"

    case2 = "請用繁體中文字寫出與\"吃飯\"，列出可能與該抽象概念有關的實例的知識圖譜"

    case3 = "請用繁體中文字列出可能與該抽象概念有關的實例的知識圖譜\n\n吃飯"

    sentences = [case1, case2, case3]
    return sentences
##@20230821

##@20230822
def get_MWP_CHN_demo_sentences__QA():
    # case1 = "根據以下<PARA>與</PARA>提供的指定文章用繁體中文字回答問題, 如果答案不在指定的文章中, 請回答\"超出文章內容無法回答\"\n\n\
    # <PARA>\n\
    # 由於個別的原因，每個家庭的成員不一定相同。\n\
    # 有些兒童與父母親同住，或是三代同堂。有些兒童只由母親或是父親獨力撫養。\n\
    # 有些兒童由祖父母、外祖父母、或其它長輩照顧。\n\
    # 三代同堂家庭是子女和父母、祖父母或外祖父母同住。\n\
    # 單親家庭是只有父親或母親負責養育小孩的家庭。\n\
    # 隔代教養家庭是由祖父母或外祖父母負責養育小孩的家庭。\n\
    # </PARA>\n\n\
    # 什麼是三代同堂?"

            # case2 = "Based on the specified article provided between <PARA> and </PARA>, answer the question in traditional Chinese characters.\n\
            # If the answer is not within the paragraph in <PARA> and </PARA>, please respond with\"超出文章內容無法回答\"\n\n\
            # <PARA>\n\
            # 由於個別的原因，每個家庭的成員不一定相同。\n\
            # 有些兒童與父母親同住，或是三代同堂。有些兒童只由母親或是父親獨力撫養。\n\
            # 有些兒童由祖父母、外祖父母、或其它長輩照顧。\n\
            # 三代同堂家庭是子女和父母、祖父母或外祖父母同住。\n\
            # 單親家庭是只有父親或母親負責養育小孩的家庭。\n\
            # 隔代教養家庭是由祖父母或外祖父母負責養育小孩的家庭。\n\
            # </PARA>\n\n\
            # 什麼是三代同堂?"
    # case2 = "根據以下<PARA>與</PARA>提供的指定文章, 並用繁體中文回答問題。\n\
    # 如果答案不在指定的文章中, 請回答\"超出文章內容無法回答\"\n\
    # 請確認你寫出的回答都是繁體中文字元。\n\n\n\
    # <PARA>\n\
    # 由於個別的原因，每個家庭的成員不一定相同。\n\
    # 有些兒童與父母親同住，或是三代同堂。有些兒童只由母親或是父親獨力撫養。\n\
    # 有些兒童由祖父母、外祖父母、或其它長輩照顧。\n\
    # 三代同堂家庭是子女和父母、祖父母或外祖父母同住。\n\
    # 單親家庭是只有父親或母親負責養育小孩的家庭。\n\
    # 隔代教養家庭是由祖父母或外祖父母負責養育小孩的家庭。\n\
    # </PARA>\n\n\
    # 我的問題是:\n\
    # 什麼是三代同堂?"        

    # case3 = "根據以下<PARA>與</PARA>提供的指定文章, 並用繁體中文回答問題。\n\
    # 如果答案不在指定的文章中, 請回答\"超出文章內容無法回答\"\n\
    # 請確認你寫出的回答都是繁體中文字元。\n\n\n\
    # <PARA>\n\
    # 由於個別的原因，每個家庭的成員不一定相同。\n\
    # 有些兒童與父母親同住，或是三代同堂。有些兒童只由母親或是父親獨力撫養。\n\
    # 有些兒童由祖父母、外祖父母、或其它長輩照顧。\n\
    # 三代同堂家庭是子女和父母、祖父母或外祖父母同住。\n\
    # 單親家庭是只有父親或母親負責養育小孩的家庭。\n\
    # 隔代教養家庭是由祖父母或外祖父母負責養育小孩的家庭。\n\
    # </PARA>\n\n\
    # 什麼是五代同堂?"    

    # sentences = [case1, case2, case3]

    case5 = "Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
    If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
    <PARA>\n\
    In this work, we survey over 180 papers from the NLP and AI communities in the field of deep learning for mathematical reasoning. \n\
    We study various types of mathematical reasoning problems, such as math word problems, theorem proving, \n\
    geometry problem solving, math question answering, and other quantitative problems\n\n\
    </PARA>\n\n\
    My question is: \n\
    How many papers do they survey in this work?"

    case6 = "Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
    If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
    <PARA>\n\
    In this work, we survey over 180 papers from the NLP and AI communities in the field of deep learning for mathematical reasoning. \n\
    We study various types of mathematical reasoning problems, such as math word problems, theorem proving, \n\
    geometry problem solving, math question answering, and other quantitative problems\n\n\
    </PARA>\n\n\
    My question is: \n\
    What food did the author eat this afternoon?"

    case7 = "Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
    If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
    <PARA>\n\
    The clinical programme includes 30 phase I trials (2 o f them conducted exclusively in patients with type 2 diabetes mellitus), \n\
    5 dose finding phase II trials (including 1 extension trial), and 13 phase IIb/III trials. \n\
    Overall, 6808 patients were treated with empagliflozin for at least 24 weeks ( 10 mg 2856 patients, 25 mg : 3943 patients),\n\
    4415 patients for at least 52 weeks ( 10 mg : 1720 patients, 25 mg : 2541 patients), and\n\
    1486 patients for at least 76 weeks ( 10 mg : 601 25 mg : 881 patients).\n\
    Most regions of the world were represented in the clinical development progra mme,\n\
    with 36.7% of the patients with diabetes coming from Europe, 31.3% from Asia, 19.7% from North America, \n\
    9.1% from Latin America, and 3.3% from Africa or the Middle East.\n\
    </PARA>\n\n\
    My question is: \n\
    How many patients with diabetes come from America?"

    case72 = "Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
    If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
    <PARA>\n\
    The clinical programme includes 30 phase I trials (2 o f them conducted exclusively in patients with type 2 diabetes mellitus), \n\
    5 dose finding phase II trials (including 1 extension trial), and 13 phase IIb/III trials. \n\
    Overall, 6808 patients were treated with empagliflozin for at least 24 weeks ( 10 mg 2856 patients, 25 mg : 3943 patients),\n\
    4415 patients for at least 52 weeks ( 10 mg : 1720 patients, 25 mg : 2541 patients), and\n\
    1486 patients for at least 76 weeks ( 10 mg : 601 25 mg : 881 patients).\n\
    Most regions of the world were represented in the clinical development progra mme,\n\
    with 36.7% of the patients with diabetes coming from Europe, 31.3% from Asia, 19.7% from North America, \n\
    9.1% from Latin America, and 3.3% from Africa or the Middle East.\n\
    </PARA>\n\n\
    My question is: \n\
    Which continents do these people come from?"

    case8 = "Answer the question based on the specified article provided between <PARA> and </PARA>, \n\
    If the answer is not within the designated article in <PARA> and </PARA>, please respond with \"The answer is beyond the content of the article.\"\n\n\
    <PARA>\n\
    The clinical programme includes 30 phase I trials (2 o f them conducted exclusively in patients with type 2 diabetes mellitus), \n\
    5 dose finding phase II trials (including 1 extension trial), and 13 phase IIb/III trials. \n\
    Overall, 6808 patients were treated with empagliflozin for at least 24 weeks ( 10 mg 2856 patients, 25 mg : 3943 patients),\n\
    4415 patients for at least 52 weeks ( 10 mg : 1720 patients, 25 mg : 2541 patients), and\n\
    1486 patients for at least 76 weeks ( 10 mg : 601 25 mg : 881 patients).\n\
    Most regions of the world were represented in the clinical development progra mme,\n\
    with 36.7% of the patients with diabetes coming from Europe, 31.3% from Asia, 19.7% from North America, \n\
    9.1% from Latin America, and 3.3% from Africa or the Middle East.\n\
    </PARA>\n\n\
    My question is: \n\
    How many phase I trials are in the clinical programme?"

    case9 = "Generate a summary in 30 words based on the specified article provided between <PARA> and </PARA>, \n\n\
    <PARA>\n\
    The clinical programme includes 30 phase I trials (2 o f them conducted exclusively in patients with type 2 diabetes mellitus), \n\
    5 dose finding phase II trials (including 1 extension trial), and 13 phase IIb/III trials. \n\
    Overall, 6808 patients were treated with empagliflozin for at least 24 weeks ( 10 mg 2856 patients, 25 mg : 3943 patients),\n\
    4415 patients for at least 52 weeks ( 10 mg : 1720 patients, 25 mg : 2541 patients), and\n\
    1486 patients for at least 76 weeks ( 10 mg : 601 25 mg : 881 patients).\n\
    Most regions of the world were represented in the clinical development progra mme,\n\
    with 36.7% of the patients with diabetes coming from Europe, 31.3% from Asia, 19.7% from North America, \n\
    9.1% from Latin America, and 3.3% from Africa or the Middle East.\n\
    </PARA>\n"

    case10 = "Generate a summary in 50 words based on the specified article provided between <PARA> and </PARA>, \n\n\
    <PARA>\n\
    The clinical programme includes 30 phase I trials (2 o f them conducted exclusively in patients with type 2 diabetes mellitus), \n\
    5 dose finding phase II trials (including 1 extension trial), and 13 phase IIb/III trials. \n\
    Overall, 6808 patients were treated with empagliflozin for at least 24 weeks ( 10 mg 2856 patients, 25 mg : 3943 patients),\n\
    4415 patients for at least 52 weeks ( 10 mg : 1720 patients, 25 mg : 2541 patients), and\n\
    1486 patients for at least 76 weeks ( 10 mg : 601 25 mg : 881 patients).\n\
    Most regions of the world were represented in the clinical development progra mme,\n\
    with 36.7% of the patients with diabetes coming from Europe, 31.3% from Asia, 19.7% from North America, \n\
    9.1% from Latin America, and 3.3% from Africa or the Middle East.\n\
    </PARA>\n"

##  @20230824
    case12 = "Generate a summary in 50 words based on the specified article provided between <PARA> and </PARA>, \n\n\
    <PARA>\n\
    This application of Keytruda (pembrolizumab 50 mg/vail) is filed by MSD Taiwan for bridging \n\
    study evaluation (BSE). Pembrolizumab is a humanized IgG4 monoclonal anti-PD-1 antibody which is \n\
    indicated for the treatment of patients with unresectable or metastatic melanoma and disease progression \n\
    following ipilimumab and, if BRAF V600 mutation positive, a BRAF inhibitor. The recommended dose \n\
    regimen of pembrolizumab is 2 mg/kg Q3W. \n\
    </PARA>\n"

    case13 = "Generate a summary in 50 words based on the specified article provided between <PARA> and </PARA>, \n\n\
    <PARA>\n\
    Cross study comparison between Japanese (PN011) and Caucasian (P001) \n\
    The pharmacokinetic of pembrolizumab in subject with advanced solid tumor has been studied in \n\
    both Japanese (PN011) and Caucasian (P001). The intensive PK sampling for non-compartment analysis \n\
    was conducted in cycle 1, and PK samplings for pre-dose and post-dose (approximately 30 min after the \n\
    start of infusion) were conducted from cycle 2. After single IV administration, the observed PK \n\
    parameters, Cmax, Vz, and AUC0-28day, in Japanese are similar to those in Caucasian at the \n\
    supratherapeutic dose, 10 mg (Table 1). Direct comparison at therapeutic dose was not available due to \n\
    lack of PK data in Non-Japanese following 2 mg The observed t1/2 value (14.1~21.6 days) relative to last \n\
    sampling time (28 days) indicated that total exposure of single dose is not completely observed. \n\
    </PARA>\n"

    case14 = "Based on the primary obvervation, generate a summary in 50 words based on the specified article provided between <PARA> and </PARA>, \n\n\
    <PARA>\n\
    Cross study comparison between Japanese (PN011) and Caucasian (P001) \n\
    The pharmacokinetic of pembrolizumab in subject with advanced solid tumor has been studied in \n\
    both Japanese (PN011) and Caucasian (P001). The intensive PK sampling for non-compartment analysis \n\
    was conducted in cycle 1, and PK samplings for pre-dose and post-dose (approximately 30 min after the \n\
    start of infusion) were conducted from cycle 2. After single IV administration, the observed PK \n\
    parameters, Cmax, Vz, and AUC0-28day, in Japanese are similar to those in Caucasian at the \n\
    supratherapeutic dose, 10 mg (Table 1). Direct comparison at therapeutic dose was not available due to \n\
    lack of PK data in Non-Japanese following 2 mg The observed t1/2 value (14.1~21.6 days) relative to last \n\
    sampling time (28 days) indicated that total exposure of single dose is not completely observed. \n\
    </PARA>\n"

    sentences = [case5, case6, case7, case72, case8, case9, case10, case12, case13, case14]

    return sentences
##@20230822

def is_pure_english_text(para):
    for myc in para:
        if not myc.isascii():
            return False
    return True

def demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=None):

##  @20230821
    if task_type=='REPHRASING':
        sentences = get_MWP_CHN_demo_sentences__rephrasing()
    elif task_type=='SENTENCE_GENERATION':
        sentences = get_MWP_CHN_demo_sentences__generating_sentences()
    elif task_type=='DIALOG_GENERATION':
        sentences = get_MWP_CHN_demo_sentences__generating_dialogs()
    elif task_type=='SOLVING_MWP':
        sentences = get_MWP_CHN_demo_sentences__solving_MWPs()
        # sentences = get_MWP_CHN_demo_sentences__solving_MWPs__2()
    elif task_type=='SOLVING_MWP_ENG':
        sentences = get_MWP_ENG_demo_sentences__solving_MWPs()
    elif task_type=='SUMMARIZATION':
        sentences = get_MWP_CHN_demo_sentences__summarization()
    elif task_type=='KG_GENERATION':
        sentences = get_MWP_CHN_demo_sentences__KGGeneration()
    elif task_type=='QA':
        sentences = get_MWP_CHN_demo_sentences__QA()
    else:
        assert False, "NOT A SUPPORTED TASK !!"
##  @20230821

    sys_prompt = None
    # print('>>load_prompt ___begin')
    # sys_prompt = load_prompt(load_chinese_prompt=USE_CHN_PROMPT)
    # print('>>load_prompt ___end')

    for query in sentences:
##      loop query, w/o history
        dialogs = []

        print('\n')
        print('>>prompt question: %s'%(query))

        if sys_prompt!=None:
            query = "%s  你要分析的句子是<S>%s</S>"%(sys_prompt, query)

##      @20230821
        if not is_pure_english_text(query):
            query = "%s  (用繁體中文回答)"%(query)
##      @20230821        
        print(query)

        dialog = [{"role": "user", "content": "%s"%(query)}]
        dialogs.append(dialog)

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
##@20230821

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
    # task_type = 'REPHRASING'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    # task_type = 'SENTENCE_GENERATION'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    # task_type = 'DIALOG_GENERATION'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    # task_type = 'SOLVING_MWP'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    # task_type = 'SUMMARIZATION'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    # task_type = 'KG_GENERATION'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')

    task_type = 'QA'
    print('*'*10)
    print('>>DEMO-TASK: %s'%(task_type))
    demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    print('')
    print('')

    # task_type = 'SOLVING_MWP_ENG'
    # print('*'*10)
    # print('>>DEMO-TASK: %s'%(task_type))
    # demo_2A_given_queries(generator, max_gen_len, temperature, top_p, task_type=task_type)
    # print('')
    # print('')
##  @20230821



if __name__ == "__main__":
    fire.Fire(main)
