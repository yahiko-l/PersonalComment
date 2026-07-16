import torch
from evaluate import load
from transformers import PreTrainedTokenizerBase
from nltk.util import ngrams
from collections import defaultdict
import spacy
from collections import OrderedDict
from transformers import GPT2Tokenizer
import jieba
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_perplexity(all_texts_list, model_id='gpt2-large',):
    """ 
    all_texts_list 输入内容为源句子
    -------------- EN --------------
    gpt2-large 

    -------------- ZH --------------
    IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese
    IDEA-CCNL/Wenzhong-GPT2-3.5B
    IDEA-CCNL/Wenzhong-GPT2-110M

    uer/gpt2-chinese-cluecorpussmall
    yuanzhoulvpi/gpt2_chinese
    mymusise/gpt2-medium-chinese
    单位：实数
    """

    """ 基于GPT的PPL评估 """
    from sdlm.metrics.perplexity import Perplexity

    temp_all_texts_list = []
    for idx, sentence in enumerate(all_texts_list):
        if len(sentence) == 0:
            sentence = 'idx'
        elif len(sentence) == 1:
            sentence += '吗'
        temp_all_texts_list.append(sentence)
    all_texts_list = temp_all_texts_list
    # torch.cuda.empty_cache()
    # perplexity = load("perplexity", module_type="metric")
    
    perplexity = Perplexity()           # 在本地执行PPL，不是从函数库执行，由于无法连接网络
    results = perplexity.compute(input_texts=all_texts_list, 
                                 add_start_token=False,
                                 model_id=model_id, 
                                 device=device,
                                 )
    result = round(results['mean_perplexity'], 2)

    return result


def compute_wordcount(all_texts_list):
    # filer the special word and char
    symbol = ['！', '@', '#', '￥', '%',  '……', '&', '*', '（', '）', '——', '-', '+',  '=', '【', '{', '】', '}', '：', '；', '“',  '”', "‘", "’", '《', '，', '》', '。', '？',  '、', '|', '、',
            '!', '@', '#', '$',  '%', '^', '&', '*', '(', ')', '_', '-',  '+', '=', '{', '[', '}', ']', ':', ';',  '"', "'", '<', ',', '>', '.', '?', '/',  '|', '\\',  ]

    words_list = []

    for sentence in all_texts_list:
        for word in sentence:
            if word not in symbol:
                words_list.append(word)

    avg_unique_num = len(set(words_list))/len(all_texts_list)
    return avg_unique_num


def compute_diversity(all_texts_list):
    """ 该计算方法存在问题，导致diversity不准确"""
    ngram_range = [2,3,4]

    token_list = []
    for sentence in all_texts_list:
        token_list.append(sentence.split(" "))
    ngram_sets = {}
    ngram_counts = defaultdict(int)

    metrics = {}
    for n in ngram_range:
        ngram_sets[n] = set()
        for tokens in token_list:
            ngram_sets[n].update(ngrams(tokens, n))
            ngram_counts[n] += len(list(ngrams(tokens, n)))
        metrics[f'{n}gram_repitition'] = (1-len(ngram_sets[n])/ngram_counts[n])

    diversity = 1
    for val in metrics.values():
        diversity *= (1-val)
    metrics['diversity'] = diversity

    return metrics


def compute_memorization(all_texts_list, human_references, n=4):
    tokenizer = spacy.load("en_core_web_sm").tokenizer
    unique_four_grams = set()
    for sentence in human_references:
        unique_four_grams.update(ngrams([str(token) for token in tokenizer(sentence)], n))

    total = 0
    duplicate = 0
    for sentence in all_texts_list:
        four_grams = list(ngrams([str(token) for token in tokenizer(sentence)], n))
        total += len(four_grams)
        for four_gram in four_grams:
            if four_gram in unique_four_grams:
                duplicate += 1

    print("memorization", duplicate/total)
    return duplicate/total


def calc_Self_BLEU(auto_comments, language):
    """
        self bleu 计算是将抽取一个 comment 剩余为 reference 句子

        auto_comments: [[comment1][comment2][comment3]]

        return: avg_score vale range [0-100]

        refs_comments 最佳参考是 生成 5条评论，其他4条评论做参考，目前将其他文章的生成评论看做参考
        SacreBLEU 直接以未分词的文本作为输入，并且对于同一个输入可以接受多个目标作为参考。
        
        单位：  %
    """
    from sacrebleu.metrics import BLEU

    if language == 'EN':
        bleu = BLEU()
    elif language == 'ZH':
        bleu = BLEU(tokenize='zh')
        # cut_comments = []
        # for auto_comment in auto_comments:
        #     auto_comment = ' '.join(jieba.cut(auto_comment, cut_all=False))
        #     cut_comments.append(auto_comment)
        # auto_comments = cut_comments
    
    score_list = []
    num = len(auto_comments) #number of sentences

    for i in range(num):
        refs_comments = [[auto_comments[j]] for j in range(num) if j!=i]
        auto_comment = [auto_comments[i]]

        score = bleu.corpus_score(auto_comment, refs_comments).score   #calcuate score
        score_list.append(score)                           #save the score

    #average score
    avg_score = round(sum(score_list) / num, 2)
    return avg_score


def calc_diversity(texts):
    """ single sentence calc distinct， 单位为 %"""
    unigram, bigram, trigram, qugram = set(), set(), set(), set()
    num_tok = 0
    for vec in texts:
        v_len = len(vec)
        num_tok += v_len
        unigram.update(vec)
        bigram.update([tuple(vec[i:i + 2]) for i in range(v_len - 1)])
        trigram.update([tuple(vec[i:i + 3]) for i in range(v_len - 2)])
        qugram.update([tuple(vec[i:i + 4]) for i in range(v_len - 3)])

    metrics = OrderedDict()
    metrics['dist_1'] = round((len(unigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_2'] = round((len(bigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_3'] = round((len(trigram) * 1.0 / num_tok) * 100, 4)
    metrics['dist_4'] = round((len(qugram) * 1.0 / num_tok) * 100, 4)

    # 暂时只显示 distin-1, -2, -3, -4
    metrics['num_d1'] = len(unigram)
    metrics['num_d2'] = len(bigram)
    metrics['num_d3'] = len(trigram)
    metrics['num_d4'] = len(qugram)

    metrics['total_num'] = num_tok
    metrics['avg_sen_len'] = round(num_tok * 1.0 / len(texts), 4)

    return metrics


def cal_diversity_n(candidate, language):
    """
    candidate = [[word1, word2, ...wordn],...,[word1, word2, ...wordm]]
    单位： %
    """
    if language == 'EN':
        candidate = [item.split(" ") for item in candidate]
    elif language == 'ZH':
        candidate = [list(jieba.cut(item)) for item in candidate]
    flatten_candidate = [si for can in candidate for si in can]
    metrics_best_n = calc_diversity(flatten_candidate)
    text_result = ','.join('{:s}={:.4f}'.format(key, metrics_best_n[key]) for key in metrics_best_n.keys())

    diversity_n = text_result.split(',')[:4]
    diversities = []

    for diversity in diversity_n:
        diversity = float(diversity.split('=')[-1])
        diversities.append(round(diversity, 2))

    return diversities


def compute_coherence(articles, gens_text, model_id):
    import torch
    from scipy.spatial.distance import cosine
    from transformers import AutoModel, AutoTokenizer
    from sklearn.metrics.pairwise import cosine_similarity

    """
    articles:
        EN: ['word1 word2', 'word1 word2'] 空格划分
        ZH: ['word1word2', 'word1word2']  原始句子，未包含分词
    gens_text:
        EN:['word1 word2', 'word1 word2'] 空格划分
        ZH:['word1word2', 'word1word2'] 原始句子，未包含分词
    -------------------- EN  ----------------------
    Model	Avg. STS
    princeton-nlp/unsup-simcse-bert-base-uncased	76.25
    princeton-nlp/unsup-simcse-bert-large-uncased	78.41
    princeton-nlp/unsup-simcse-roberta-base	76.57
    princeton-nlp/unsup-simcse-roberta-large	78.90
    princeton-nlp/sup-simcse-bert-base-uncased	81.57
    princeton-nlp/sup-simcse-bert-large-uncased	82.21
    princeton-nlp/sup-simcse-roberta-base	82.52
    princeton-nlp/sup-simcse-roberta-large	83.76   # 英文当前采用的语言模型

    -------------------- ZH  ----------------------
    liusiyi641/simcse_unsup_chinese_bert_cnews
    IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese
    cyclone/simcse-chinese-roberta-wwm-ext
    
    单位 %
    """

    # Import our models. The package will take care of downloading the models automatically
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)

    soces = []

    for article, gen_text in zip(articles, gens_text):
        # Tokenize input texts
        article = article[:510]
        texts = [article, gen_text]

        """ 单独计算获得 Embedding 表示 """
        inputs_a = tokenizer(texts[0], return_tensors="pt").to(device)
        inputs_b = tokenizer(texts[1], return_tensors="pt").to(device)
        outputs_a = model(**inputs_a ,output_hidden_states=True)
        texta_embedding = outputs_a.hidden_states[-1][:,0,:].squeeze()
        outputs_b = model(**inputs_b ,output_hidden_states=True)
        textb_embedding = outputs_b.hidden_states[-1][:,0,:].squeeze()
        silimarity_soce = cosine_similarity(texta_embedding.reshape(1,-1).cpu().detach().numpy(), textb_embedding.reshape(1,-1).cpu().detach().numpy())[0][0]

        """ 一起计算获得  Embedding 表示"""
        # inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        # # Get the embeddings
        # with torch.no_grad():
        #     embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        # # Calculate cosine similarities
        # # Cosine similarities are in [-1, 1]. Higher means more similar, and then Normalize the value of cosine
        # silimarity_soce = 1 - cosine(embeddings[0].cpu(), embeddings[1].cpu())

        silimarity_soce = 0.5 + 0.5 * silimarity_soce     # cosine normalization
        soces.append(silimarity_soce)

    avg_soces = round(sum(soces) / len(soces) * 100, 2)
    return avg_soces


def compute_bert_score(articles, gens_text, language, bertscore_id):
    """
    bert-base-chinese
    """
    language = language.lower()
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=gens_text, references=articles, lang=language, device=device)
    # results = bertscore.compute(predictions=gens_text, references=articles, model_type=bertscore_id, device=device)
    bert_scores = round(np.mean(results['f1'])*100, 2)
    return bert_scores


def compute_avg_word_length(all_texts_list):
    words_list = []
    total_sentence_len = 0
    for sentence in all_texts_list:
        total_sentence_len += len(sentence)
    
    avg_sentence_len = round(total_sentence_len / len(all_texts_list), 2)
    return avg_sentence_len