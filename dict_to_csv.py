import pandas as pd
import json
import argparse


def data_save(data, path):
    json.dump(data, open(path , 'w', encoding='utf-8'), indent=4, ensure_ascii=False)

 
def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def main(path=None):
    results = data_read(path)
    diversity_n = results['diversity_n']

    for idx, dist_i in enumerate(diversity_n):
        results[f"dist_{idx+1}"] = dist_i
    del results['diversity_n']

    results = [results]

    df = pd.DataFrame(results)

    saving_path = '/'.join(path.split('/')[:-1]) + "/evaluation.csv"
    df.to_csv(saving_path, index=False, header=True)
    print(saving_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert evaluation JSON to CSV.")
    parser.add_argument("path", help="Path to evaluation.json")
    main(path=parser.parse_args().path)
