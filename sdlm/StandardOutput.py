import json


def data_save(data, path):
    json.dump(data, open(path , 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def standardprocessing(path):
    data = data_read(path)
    src_inputs = data['src_input'] 
    gold_texts_maskeds = data['gold_texts_masked']
    prefixes = data['prefixes']
    pred_texts_from_logits_maskeds = data['pred_texts_from_logits_masked']
    
    samples = []
    for  src, gold_text, prefixe,  pred in zip(src_inputs, gold_texts_maskeds, prefixes, pred_texts_from_logits_maskeds):
        sample = {
            "article": src,
            "ground_truth_sentence":gold_text,
            "peronsal":prefixe,
            "generated_sentence":pred,
        }

        samples.append(sample)
    
    saving_path = '/'.join(path.split('/')[:-1]) + '/StandardOutput_one.json'
    data_save(samples, saving_path)
    print(saving_path)

    from evaluation import main as evaluation
    evaluation(saving_path)


if __name__ == "__main__":
    path = "outputs/comment_roberta-seq2seq_PersonaComment_train_2023-10-09-10:15_self_condition_K_5/all_results.json"
    standardprocessing(path)


