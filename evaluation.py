import json
import yaml
import jieba
import argparse
import os

from sdlm.metrics.metric import compute_perplexity, compute_wordcount, \
compute_diversity, compute_memorization, calc_Self_BLEU, \
cal_diversity_n, compute_coherence, compute_bert_score, compute_avg_word_length


"""
通过json文件计算自动评估指标
"""


def data_save(data, path):
    json.dump(data, open(path , 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

 
def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def main():
    parser = argparse.ArgumentParser(description="评估参数")
    # 添加参数
    parser.add_argument("path", help="Generation-results JSON file")
    parser.add_argument(
        "--skip-attribute-classifiers",
        action="store_true",
        help="Skip gender, age and location accuracy (no classifier checkpoints required).",
    )

    # 解析参数
    args = parser.parse_args()
    path = args.path

    gens_text = []
    refs_text = []
    articles = []
    logs = {}

    """ the generation data from DiffuPercom model and baseline """
    save_evaluation_file = '/'.join(path.split('/')[:-1]) + '/evaluation.json'
    gen_data_dict = data_read(path)

    for item in gen_data_dict:
        article = ''.join(item['article'].split(" "))
        ground_truth_sentence = ''.join(item['ground_truth_sentence'].split(" "))
        generated_sentence = ''.join(item['generated_sentence'].split(" "))
        language = 'ZH'
        
        articles.append(article)
        refs_text.append(ground_truth_sentence)
        gens_text.append(generated_sentence)

    print(language)

    model_id = "mymusise/gpt2-medium-chinese"
    coherence_id = "cyclone/simcse-chinese-roberta-wwm-ext"
    bertscore_id = "bert-base-chinese"

    coherence_score = compute_coherence(articles, gens_text, coherence_id)
    logs.update({'coherence_score': coherence_score})
    print("coherence_score:", coherence_score)

    bert_scores = compute_bert_score(articles, gens_text, language, bertscore_id)
    logs.update({'bert_scores': bert_scores})
    print("bert_score:", bert_scores) 

    perplexity = compute_perplexity(gens_text, model_id=model_id)
    logs.update({'perplexity': perplexity})
    print("perplexity:", perplexity)

    self_bleu = calc_Self_BLEU(gens_text, language=language)
    logs.update({'self_bleu': self_bleu})
    print("self_bleu:", self_bleu)

    diversity_n = cal_diversity_n(gens_text, language=language)
    logs.update({'diversity_n': diversity_n})
    print("diversity_n:", diversity_n)

    wordcount = compute_wordcount(gens_text)
    logs.update({'wordcount': wordcount})
    print("arg_wordcount:", wordcount)
    
    avg_sentence_len = compute_avg_word_length(gens_text)
    logs.update({'avg_sentence_length': avg_sentence_len})
    print("avg_wordcount:", avg_sentence_len)

    if not args.skip_attribute_classifiers:
        raise ValueError(
            "Attribute evaluation needs the paper's classifier checkpoints. "
            "Pass --skip-attribute-classifiers for text-only metrics."
        )

    data_save(logs, save_evaluation_file)
    print(save_evaluation_file)

    # from dict_to_csv import main
    # main(path=save_evaluation_file)


if __name__ == "__main__":
    main()
