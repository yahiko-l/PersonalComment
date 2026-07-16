import os
import re
import json
import pickle
from functools import reduce
from copy import deepcopy
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
from collections import Counter


def data_save(data, path):
    json.dump(data, open(path , 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


def get_cache_path(*args):
    key = "-".join(list([str(i) for i in args]))
    sub_key = re.sub(r"{}".format(os.sep), '-', key)
    prefix = '/'.join(sub_key.split('-')[:2])
    postfix = '-'.join(sub_key.split('-')[2:])
    cache_path = "{}/cache/{}".format(prefix, postfix)
    return cache_path


def retrieve_cache(*args, **kwargs):
    if kwargs != {}:
        raise ValueError("Not implement!")
    cache_path = get_cache_path(*args)
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as file:
            cache_obj = pickle.load(file)
            return cache_obj
    return None


def save_as_cache(cache_object, *args, **kwargs):
    if kwargs != {}:
        raise ValueError("Not implement!")
    cache_path = get_cache_path(*args)
    os.makedirs('cache', exist_ok=True)
    with open(cache_path, 'wb') as file:
        pickle.dump(cache_object, file)
        return cache_object


class PersonalCommentDataset_ML(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path,
                 tokenizer,
                 data_args=None,
                 ):
        
        self.path = data_path
        self.data_args = data_args
        self.limit_length = data_args.limit_length
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.gender_dict = {'f':'女', 'm':'男',}
        self.label2id = {"负面": 0, "中性": 1, "正面": 2}

        self.comments_list = self.get_comments(path=data_path)
        self.articles_dict = self.get_articles(path=data_path)

    def get_comments(self, path):
        comments_file_path = os.path.join(path, 'comments.data')
        comments_list = data_read(comments_file_path)

        count = 0
        comments = []
        for comment in comments_list:
            if self.limit_length is not None and count == self.limit_length:
                break
            count += 1

            _id = comment['_id']
            content = ''.join(comment['content'].split(' '))

            persona = comment['userinfo']
            gender = self.gender_dict[persona['gender']]
            location = persona['location']
            description = ''.join(persona['description'].split(' '))
            age = persona['age']

            sentiment = comment['sentiment']
            sentiment = self.label2id[sentiment]

            persona = (f"性别:{gender};年龄:{age};位置:{location};个人描述:{description}")

            content_input = self.tokenizer(content,
                                            add_special_tokens=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=False,
                                            truncation=True,
                                            max_length=self.data_args.max_comment_len,)

            persona_input = self.tokenizer(persona,
                                            add_special_tokens=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=False,
                                            truncation=True,
                                            max_length=self.data_args.max_persona_len,)            

            data = {'_id':_id,
                    'content':content_input['input_ids'],
                    'persona':persona_input['input_ids'],
                    'sentiment': sentiment}
            
            comments.append(data)

        return comments

    def tokenize_function(self, examples):
        comments = examples['comments']
        personas = examples['personas']

        comments_input = self.tokenizer(comments,
                                        add_special_tokens=True,
                                        max_length=self.data_args.max_comment_len,)

        personas_input = self.tokenizer(personas,
                                        add_special_tokens=True,
                                        max_length=self.data_args.max_persona_len,)

        result_dict = {'comments': comments_input, 'personas': personas_input}
        return result_dict

    def get_articles(self, path):
        articles_file_path = os.path.join(path, 'articles.data')
        articles_dict = data_read(articles_file_path)

        new_articles_dict = {}
        for key, value in articles_dict.items():
            article = self.article_body_concat(value)

            article_input = self.tokenizer(article,
                                           add_special_tokens=True,
                                           return_token_type_ids=False,
                                           return_attention_mask=False,
                                           truncation=True,
                                           max_length=self.data_args.max_article_len,)      

            new_articles_dict.update({key: article_input['input_ids']})
            
        return new_articles_dict

    def article_body_concat(self, article):
        title = article['title']
        content = ' '.join(article['content'])
        article = ' '.join([title, content])
        article = ''.join(article.split(' '))
        return article

    def __len__(self):
        return len(self.comments_list)

    def __getitem__(self, idx):
        batch = self.comments_list[idx]
        _id = batch['_id']
        article = self.articles_dict[_id]
        commnet = batch['content']
        persona = batch['persona']
        sentiment = batch['sentiment']

        return {'persona': persona,
                'comment': commnet,
                'article': article,
                'sentiment': sentiment}


def collate_fn(sample_list):
    to_be_flattened = ['persona', 'comment', 'article', 'sentiment']

    data = {}
    for key in to_be_flattened:
        if key not in sample_list[0].keys():
            continue
        if sample_list[0][key] is None:
            continue
        flatten_samples = [sample[key] for sample in sample_list]
        data[key] = flatten_samples

    return data


def collate_fn_with_ML_tokenizer(tokenizer, config):

    def build_collate_fn(sample_list):
        data = collate_fn(sample_list)
        # Because we will discard [CLS] in targets, so we add max_length by 1
        persona_input = tokenizer.pad(data['persona'], 
                                       return_tensors='pt', 
                                       padding='max_length',
                                       max_length=config.max_persona_len,)
        
        src_input = tokenizer.pad(data['article'], 
                                   return_tensors='pt', 
                                   padding='max_length',
                                   max_length=config.max_article_len,)

        tgt_input = tokenizer.pad(data['comment'], 
                              return_tensors='pt', 
                              padding='max_length',
                              max_length=config.max_comment_len,)
           
        del data['persona']
        del data['article']
        del data['comment']
        
        # data['persona_input'] = persona_input
        # data['query_input'] = src_input
        data['input_ids'] = tgt_input['input_ids']
        # data['persona_query_input'] = persona_article_input
        return data

    return build_collate_fn


class PersonalCommentDataset_Seq2seq(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path,
                 tokenizer,
                 data_args=None,
                 ):
        
        self.path = data_path
        self.data_args = data_args
        self.limit_length = data_args.limit_length
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.gender_dict = {'f':'女', 'm':'男',}

        self.comments_list = self.get_comments(path=data_path)
        self.articles_dict = self.get_articles(path=data_path)

    def get_comments(self, path):
        comments_file_path = os.path.join(path, 'comments.data')
        comments_list = data_read(comments_file_path)

        count = 0
        comments = []
        for comment in comments_list:
            if self.limit_length is not None and count == self.limit_length:
                break
            count += 1

            _id = comment['_id']
            content = ''.join(comment['content'].split(' '))

            persona = comment['userinfo']
            gender = self.gender_dict[persona['gender']]
            location = persona['location']
            description = ''.join(persona['description'].split(' '))
            age = persona['age']
            
            persona = (f"性别:{gender};年龄:{age};位置:{location};个人描述:{description}")

            content_input = self.tokenizer(content,
                                            add_special_tokens=True,
                                            max_length=self.data_args.max_comment_len,
                                            padding=False,
                                            truncation=True,                                            
                                            )

            persona_input = self.tokenizer(persona,
                                            add_special_tokens=True,
                                            max_length=self.data_args.max_persona_len,
                                            padding=False,
                                            truncation=True,                                             
                                            )            

            data = {'_id':_id,
                    'content':content_input,
                    'persona':persona_input,}
            
            comments.append(data)

        return comments

    def get_articles(self, path):
        articles_file_path = os.path.join(path, 'articles.data')
        articles_dict = data_read(articles_file_path)

        new_articles_dict = {}
        for key, value in articles_dict.items():
            article = self.article_body_concat(value)

            article_input = self.tokenizer(article,
                                           add_special_tokens=True,
                                           max_length=self.data_args.max_article_len,
                                           padding=False,
                                           truncation=True,
                                           )

            new_articles_dict.update({key: article_input})
            
        return new_articles_dict

    def article_body_concat(self, article):
        title = article['title']
        content = ' '.join(article['content'])
        article = ' '.join([title, content])
        article = ''.join(article.split(' '))
        return article

    def __len__(self):
        return len(self.comments_list)

    def __getitem__(self, idx):
        batch = self.comments_list[idx]
        _id = batch['_id']
        article = self.articles_dict[_id]
        commnet = batch['content']
        persona = batch['persona']

        return {'persona': persona,
                'comment': commnet,
                'article': article,}


def collate_fn_with_Seq2seq_tokenizer(tokenizer, config):

    def build_collate_fn(sample_list):
        """ training_data_type = Seq2seq, input_id = src_input+tgt_input """
        data = collate_fn(sample_list)
       
        input_ids = data['article']
        labels = data['comment']

        input_target = [source['input_ids'] + target['input_ids'] for source, target in zip(input_ids, labels)]
        features = tokenizer.pad(
            {"input_ids": input_target},
            padding='max_length',
            max_length=config.max_seq_length,
            return_tensors="pt",
        )

        batch_length = features["input_ids"].shape[1]
        masks = [
            len(input_id['input_ids'] ) * [False] + (batch_length - len(input_id['input_ids'] )) * [True]
            for input_id in input_ids
        ]
        features["span_mask"] = torch.tensor(masks)
        
        return features

    return build_collate_fn


def collate_fn_with_Comment_tokenizer(tokenizer, config):

    def build_collate_fn(sample_list):
        """ training_data_type = PersonaComment, input_id = person_input+tgt_input src_input person_input  is_low_memory=False"""
        data = collate_fn(sample_list)
        persona_input = tokenizer.pad({"input_ids": data['persona']}, 
                                       return_tensors='pt', 
                                       padding='max_length',
                                       max_length=config.max_persona_len,)
        
        src_input = tokenizer.pad({"input_ids": data['article']}, 
                                   return_tensors='pt', 
                                   padding='max_length',
                                   max_length=config.max_article_len,)

        personas = data['persona']
        comments = data['comment']
        input_target = [persona + comment for persona, comment in zip(personas, comments)]
        features = tokenizer.pad(
            {"input_ids": input_target},
            padding='max_length',
            max_length=config.max_persona_len + config.max_comment_len,
            return_tensors="pt",
        )

        batch_length = features["input_ids"].shape[1]
        masks = [
            len(input_id) * [False] + (batch_length - len(input_id)) * [True]
            for input_id in personas
        ]
        features["span_mask"] = torch.tensor(masks)

        features["src_input"] = src_input
        features["persona_input"] = persona_input

        features["sentiment"] = data['sentiment']

        return features

    return build_collate_fn



""" classifier collater """
def infinite_loader(data_loader):
    while True:
        yield from data_loader


class PersonalCommentDataset_Classifier(torch.utils.data.Dataset):
    def __init__(self, 
                 data_path,
                 tokenizer,
                 data_args=None,
                 ):
        
        self.path = data_path
        self.data_args = data_args
        self.limit_length = data_args.limit_length
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.label2id = {"负面": 0, "正面": 1, "中性": 2,}
        self.gender2id = {'f': 0, 'm': 1}

        self.comments_list = self.get_comments(path=data_path)
        # self.age_statistics(self.comments_list)
        # self.location_statistics(self.comments_list)

    def get_comments(self, path):
        comments_file_path = os.path.join(path, 'comments.data')
        comments_list = data_read(comments_file_path)

        count = 0
        comments = []
        sentiments = []
        for comment in comments_list:
            if self.limit_length is not None and count == self.limit_length:
                break
            count += 1

            _id = comment['_id']
            content = ''.join(comment['content'].split(' '))

            persona = comment['userinfo']
            age = self.age_mapping(persona['age'])

            gender = self.gender2id[persona['gender']]
            location = self.location_mapping(persona['location'])

            sentiment = comment['sentiment']
            sentiment = self.label2id[sentiment]

            # removing the neutrality label, it will affect the results generated by the classifier guidance.
            if sentiment == 2:
                continue
            
            assert sentiment != 2

            content_input = self.tokenizer(content,
                                            add_special_tokens=True,
                                            return_token_type_ids=False,
                                            return_attention_mask=False,
                                            truncation=True,
                                            max_length=self.data_args.max_comment_len,)    

            data = {'_id':_id,
                    'content':content_input['input_ids'],
                    'age':age,
                    'gender':gender,
                    'location':location,
                    'sentiment': sentiment}
            
            comments.append(data)
            sentiments.append(sentiment)

        print(Counter(sentiments))
        return comments

    def age_statistics(self, comments_list):
        """
            post_70s:183251 
            post_80s:135509 
            post_90s:154369 
            post_00s:99895
        """
        post_70s = 0
        post_80s = 0
        post_90s = 0
        post_00s = 0 

        for item in comments_list:
            age = item['age']

            if age >= 44:
                post_70s += 1
            if age >=34 and age <= 43:
                post_80s += 1
            if age >= 24 and age <= 33:
                post_90s += 1
            if age <= 23:
                post_00s += 1

        print(post_70s, post_80s, post_90s, post_00s)
        print()

    def age_mapping(self, age):
        if age >= 44:
            age = 0
        elif age >=34 and age <= 43:
            age = 1
        elif age >= 24 and age <= 33:
            age = 2
        elif age <= 23:
            age = 3

        return age

    def location_statistics(self, comments_list):
        locations = []
        for item in comments_list:
            location = item['location']
            locations.append(location)
        
        print(Counter(locations))
        print()

    def location_mapping(self, location):       
        NorthChina = ['北京', '天津', '河北', '山西', '内蒙古']
        NortheastChina = ['辽宁', '吉林', '黑龙江']
        EastChina = ['上海', '江苏', '浙江', '安徽', '福建', '江西', '山东', '台湾']
        CentralSouthChina = ['河南', '湖北', '湖南', '广东', '广西', '海南', '香港', '澳门']
        SouthwestChina = ['重庆', '四川', '贵州', '云南', '西藏']
        NorthwestChina = ['陕西', '甘肃', '青海', '宁夏', '新疆']
        Other = ['其他', '海外']

        if location in NorthChina:
            location = 0
        elif location in NortheastChina:
            location = 1
        elif location in EastChina:
            location = 2
        elif location in CentralSouthChina:
            location = 3
        elif location in SouthwestChina:
            location = 4
        elif location in NorthwestChina:
            location = 5
        elif location in Other:
            location = 6
        else:
            print(location)

        return location

    def __len__(self):
        return len(self.comments_list)

    def __getitem__(self, idx):
        batch = self.comments_list[idx]
        comment = batch['content']
        sentiment = batch['sentiment']
        age = batch['age']
        gender = batch['gender']
        location = batch['location']
        
        comment_input = self.tokenizer.pad({"input_ids": comment}, 
                                       return_tensors='pt', 
                                       padding='max_length',
                                       max_length=self.data_args.max_comment_len,)
        
        return {'comment': comment_input,
                'sentiment': sentiment,
                'age': age,
                'gender': gender,
                'location': location,}


""" update dataset loading for PersonalComment"""
def paddding_token(data, tokenizer, max_length):
    new_data = []
    for item in data:
        pad_content = (max_length - len(item)) * [tokenizer.pad_token_id]
        new_item = item + pad_content
        new_data.append(new_item)
    
    return new_data


def collate_fn_with_Comment_tokenizer_Update(tokenizer, config):
    """
    Modify the previous sequence of persona + comment + padding to persona + padding + comment + padding.
    Due to the shorter padding length of comments and the reduced number of padding, the cross-entropy loss increases.
    {'loss': 2.4463, 'learning_rate': 1e-05, 'epoch': 5.13}                                                                                                                
    {'loss': 2.1321, 'learning_rate': 1e-05, 'epoch': 5.16}                                                                                                                
    {'loss': 2.3201, 'learning_rate': 1e-05, 'epoch': 5.19}                                                                                                                
    {'loss': 2.2655, 'learning_rate': 1e-05, 'epoch': 5.23}                                                                                                                
    {'loss': 2.276, 'learning_rate': 1e-05, 'epoch': 5.26}                                                                                                                 
    {'loss': 2.3674, 'learning_rate': 1e-05, 'epoch': 5.3}                                                                                                                 
    {'loss': 2.1343, 'learning_rate': 1e-05, 'epoch': 5.33}                                                                                                                
    {'loss': 2.2532, 'learning_rate': 1e-05, 'epoch': 5.36}     

    Don't use the method
    """
    def build_collate_fn(sample_list):
        """ training_data_type = PersonaComment, input_id = person_input+tgt_input src_input person_input  is_low_memory=False"""
        data = collate_fn(sample_list)
        persona_input = tokenizer.pad({"input_ids": data['persona']}, 
                                       return_tensors='pt', 
                                       padding='max_length',
                                       max_length=config.max_persona_len,)
        
        src_input = tokenizer.pad({"input_ids": data['article']}, 
                                   return_tensors='pt', 
                                   padding='max_length',
                                   max_length=config.max_article_len,)

        personas = data['persona']
        personas = paddding_token(personas, tokenizer, config.max_persona_len)

        comments = data['comment']
        input_target = [persona + comment for persona, comment in zip(personas, comments)]
        features = tokenizer.pad(
            {"input_ids": input_target},
            padding='max_length',
            max_length=config.max_persona_len + config.max_comment_len,
            return_tensors="pt",
        )

        batch_length = features["input_ids"].shape[1]
        masks = [
            len(input_id) * [False] + (batch_length - len(input_id)) * [True]
            for input_id in personas
        ]
        features["span_mask"] = torch.tensor(masks)

        features["src_input"] = src_input
        features["persona_input"] = persona_input

        features["sentiment"] = data['sentiment']

        return features

    return build_collate_fn

