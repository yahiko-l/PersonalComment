"""
train the classifier base on the pretrain model

"num_labels": 7
"""

import logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import sys
import time
import shutil
import json

import wandb
os.environ.setdefault("WANDB_MODE", "offline")
import torch
import datasets
import evaluate
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers import BertForSequenceClassification, AutoModelForSequenceClassification
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from torch.optim import AdamW

from sdlm.arguments import get_summarization_args

from sdlm.data.dataset import PersonalCommentDataset_Classifier as PersonalCommentDataset_Seq2seq
from torch.utils.data import DataLoader
from sdlm.data.dataset import infinite_loader
from torch import nn

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")
require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits):
    return logits.argmax(dim=-1)


def main(mode='train', generated_data_path=None, checkpoint_path=None, config_path='configs/classifier/location.json'):
    model_path = config_path
    model_args, data_args, training_args, _ = get_summarization_args(model_path)
    pretrained_path = checkpoint_path

    if mode == 'train':
        output_dir = training_args.output_dir
        training_args.output_dir = '_'.join([output_dir, training_args.training_data_type, data_args.train_file.split('/')[-1], time.strftime("%Y-%m-%d-%H:%M", time.localtime())])
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        shutil.copy(model_path, training_args.output_dir)

    if training_args.training_data_type == "Seq2seq":
        assert (
            data_args.max_article_len + data_args.max_comment_len <= data_args.max_seq_length
        )
    else:
        assert(data_args.max_article_len <= data_args.max_seq_length)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    log_level = 20
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    # define model
    logger.info("loding the model")

    """ Transfomers """    
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path)
    classifier = nn.Linear(model.bert.config.hidden_size, model_args.num_labels)
    model.classifier = classifier
    model.num_labels = model_args.num_labels

    # We resize the embeddings only when necessary to avoid index errors. 
    # If you are creating a model from scratch on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    """ CNN or RNN """
    vocab_size = len(tokenizer)

    # from classifier.models.CNN import CNN
    # model = CNN(batch_size=training_args.per_device_train_batch_size, 
    #             output_size=model_args.num_labels, 
    #             in_channels=1, 
    #             out_channels=100, 
    #             kernel_heights=[3,4,5], 
    #             keep_probab=0.5, 
    #             vocab_size=vocab_size, 
    #             embedding_length=300, 
    #             weights=None)

    # from classifier.models.LSTM import LSTMClassifier
    # model = LSTMClassifier(batch_size=training_args.per_device_train_batch_size, 
    #                         output_size=model_args.num_labels, 
    #                         hidden_size=300, 
    #                         vocab_size=vocab_size, 
    #                         embedding_length=300, 
    #                         weights=None,)

    # from classifier.models.RCNN import RCNN
    # model = RCNN(batch_size=training_args.per_device_train_batch_size, 
    #              output_size=model_args.num_labels, 
    #              hidden_size=300, 
    #              vocab_size=vocab_size, 
    #              embedding_length=300, 
    #              weights=None)

    # from classifier.models.LSTM_Attn import AttentionModel
    # model = AttentionModel(batch_size=training_args.per_device_train_batch_size, 
    #                        output_size=model_args.num_labels, 
    #                        hidden_size=300, 
    #                        vocab_size=vocab_size, 
    #                        embedding_length=300, 
    #                        weights=None)


    """ Initialize dataset & dataloader """
    train_dataset = PersonalCommentDataset_Seq2seq(data_args.train_file,
                                                   tokenizer,
                                                   data_args=data_args,)
    train_dataloader = DataLoader(train_dataset,
                                batch_size=training_args.per_device_train_batch_size,
                                shuffle=True,
                                num_workers=training_args.dataloader_num_workers,)
    train_dataloader = infinite_loader(train_dataloader)


    eval_dataset = PersonalCommentDataset_Seq2seq(data_args.test_file,
                                                  tokenizer,
                                                  data_args=data_args,)
    eval_dataloader = DataLoader(eval_dataset,
                                batch_size=training_args.per_device_eval_batch_size,
                                shuffle=False,
                                num_workers=training_args.dataloader_num_workers,) 

    model.to(training_args.device)

    pytorch_total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"the parameter count is {pytorch_total_params}")

    # -------------------->
    resume_step = 0

    """ loding the checkpoint """
    if mode == "inference":
        if not pretrained_path:
            raise ValueError("--checkpoint is required in inference mode")
        print(f"loading model from checkpoint: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=training_args.device)
        model.load_state_dict(checkpoint)

    logger.info(f"creating optimizer...")
    optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    # if training_args.resume_from_checkpoint:
    #     # loading the optimizer parameter
    #     opt_checkpoint = bf.join(bf.dirname(args.resume_checkpoint), f"opt{resume_step:06}.pt")
    #     logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
    #     optimizer.load_state_dict(dist_util.load_state_dict(opt_checkpoint, map_location=dist_util.dev()))

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "Classifier"),
        name=training_args.output_dir,
    )

    logger.info("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        optimizer.zero_grad()
        batch = next(data_loader)
        inputs = batch["comment"].to(training_args.device)
        labels = batch["location"].to(training_args.device)

        # In the config of pretrain_models, 
        # add num_labels and ensure that num_labels is consistent with the number of gender labels.
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        losses = {}
        losses[f"{prefix}_loss"] = loss.item()
        losses[f"{prefix}_acc@1"] = compute_top_k(
            logits, labels, k=1, reduction="mean"
        )

        loss = loss.mean()
        loss.backward()
        optimizer.step()

        return losses

    def evaluation(data_loader, prefix="val"):
        total_loss = []
        total_acc= []

        optimizer.zero_grad()
        for idx, batch in enumerate(data_loader):
            inputs = batch["comment"].to(training_args.device)
            labels = batch["location"].to(training_args.device)

            # In the config of pretrain_models, 
            # add num_labels and ensure that num_labels is consistent with the number of gender labels.
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss.append(loss.item())
            acc = compute_top_k(logits, labels, k=1, reduction="mean")
            total_acc.append(acc)
            
        losses = {}
        losses[f"{prefix}_loss"] = sum(total_loss)/len(total_loss)
        losses[f"{prefix}_acc@1"] = sum(total_acc)/len(total_acc)
        return losses

    if mode == 'train':
        for step in range(training_args.max_steps - resume_step):
            logger.info("step: %s", step + resume_step)
            logger.info("samples: %s", (step + resume_step + 1) * training_args.per_device_train_batch_size)

            # training_args.anneal_lr = False
            # if training_args.anneal_lr:
            #     set_annealed_lr(optimizer, training_args.learning_rate, (step + resume_step) / training_args.max_steps)

            train_losses = forward_backward_log(train_dataloader, prefix="train")

            if eval_dataloader is not None and step != 0 and not step % training_args.eval_steps:
                with torch.no_grad():
                    model.eval()
                    eval_losses = evaluation(eval_dataloader, prefix="valid")
                    wandb.log(eval_losses)
                    print(eval_losses)
                model.train()

            if not step % training_args.logging_steps:
                train_losses.update({"step": step})
                wandb.log(train_losses)
                print(train_losses)

            if (step and not (step + resume_step) % training_args.save_steps):
                logger.info("saving model...")
                save_model(training_args, model, tokenizer, optimizer, step + resume_step)
                # save_model_transfomer(training_args, model, tokenizer, optimizer, step + resume_step)
    elif mode == 'inference':
        assert generated_data_path != None
        generated_data = data_read(generated_data_path)
        
        total_acc = []
        for item in generated_data:
            _, _, location = attribute_parse(item["personal"], generated_data_path)

            generated_sentence = tokenizer(item["generated_sentence"],
                                    add_special_tokens=True,
                                    return_token_type_ids=False,
                                    return_attention_mask=False,
                                    return_tensors='pt',)

            inputs = generated_sentence.to(training_args.device)
            labels = torch.tensor([location]).to(training_args.device)
            
            outputs = model(**inputs, labels=labels)
            logits = outputs.logits

            acc = compute_top_k(logits, labels, k=1, reduction="mean")
            total_acc.append(acc)
        
        avg_acc = sum(total_acc) / len(total_acc)
        print(avg_acc)

        return avg_acc

    logger.info("saving model...")
    save_model(training_args, model, tokenizer, optimizer, step + resume_step)
    # save_model_transfomer(training_args, model, tokenizer, optimizer, step + resume_step)


def set_annealed_lr(opt, base_lr, frac_done):
    lr = base_lr * (1 - frac_done)
    for param_group in opt.param_groups:
        param_group["lr"] = lr


def save_model_transfomer(args, model, tokenizer, opt, step):
    checkpoint_path = os.path.join(args.output_dir, f"step_{step:06d}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model.save_pretrained(checkpoint_path)
    torch.save(opt.state_dict(), os.path.join(checkpoint_path, f"opt.pt"))
    tokenizer.save_pretrained(checkpoint_path)


def save_model(args, model, tokenizer, opt, step):
    checkpoint_path = os.path.join(args.output_dir, f"step_{step:06d}")
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    torch.save(opt.state_dict(), os.path.join(checkpoint_path, f"opt.pt"))
    torch.save(model.state_dict(), os.path.join(checkpoint_path, f"model.pt"))
    tokenizer.save_pretrained(checkpoint_path)


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = torch.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def attribute_parse(attribute, generated_data_path):
    # personal_type_parse
    if "Baselines" in generated_data_path:
        gender = attribute.split("性别")[1]
        gender = gender.split('年龄')[0]

        age = attribute.split("年龄")[1]
        age = age.split('地址')[0]

        location = attribute.split("地址")[1]
        location = location.split('个人描述')[0]
    else:
        attribute = attribute.split(";")
        gender, age, location = attribute[0], attribute[1], attribute[2]

        gender = gender.split(":")[-1]
        age = age.split(":")[-1]
        location = location.split(":")[-1]

    gender2id = {'女': 0, '男': 1}
    gender = gender2id[gender]

    age = age_mapping(int(age))

    location = location_mapping(location)

    return gender, age, location


def age_mapping(age):
    if age >= 44:
        age = 0
    elif age >=34 and age <= 43:
        age = 1
    elif age >= 24 and age <= 33:
        age = 2
    elif age <= 23:
        age = 3

    return age


def location_mapping(location):       
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
        location = 6

    return location


def data_save(data, path):
    json.dump(data, open(path , 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


def data_read(path):
    with open(path, 'r', encoding='utf-8') as load_f:
        load_dict = json.load(load_f)
    return load_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or run the location classifier.")
    parser.add_argument("--mode", choices=("train", "inference"), default="train")
    parser.add_argument("--config", default="configs/classifier/location.json")
    parser.add_argument("--generated-data")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    main(args.mode, args.generated_data, args.checkpoint, args.config)
