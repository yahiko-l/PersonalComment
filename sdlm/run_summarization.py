"""
Fine-tuning the library models for sequence to sequence.
"""

import logging
import os
import argparse
import pdb
import sys
import time
import shutil
import wandb
os.environ.setdefault("WANDB_MODE", "offline")

import datasets
import evaluate
import nltk
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from sdlm.data.postprocessors import postprocess_text_for_metric
from sdlm.inference.inference_utils import process_text

from sdlm.schedulers import SimplexDDPMScheduler
from sdlm.trainer import DiffusionTrainer
from sdlm.arguments import get_summarization_args


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0")
require_version("datasets>=1.8.0")
logger = logging.getLogger(__name__)


def preprocess_logits_for_metrics(logits):
    return logits.argmax(dim=-1)


def main(model_path):
    model_args, data_args, training_args, diffusion_args = get_summarization_args(model_path)

    print(data_args)
    if training_args.do_train:
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

    wandb.init(
        project=os.getenv("WANDB_PROJECT", data_args.dataset_name),
        name=training_args.output_dir,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

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

    if "roberta" in model_args.model_name_or_path.lower():
        if training_args.training_data_type == "Seq2seq":
            from sdlm.data.dataset import PersonalCommentDataset_ML as PersonalCommentDataset_Seq2seq
            from sdlm.data.dataset import collate_fn_with_Seq2seq_tokenizer as data_collator
            from sdlm.models import RobertaDiffusionConfigLM, RobertaForDiffusionLM
            ModelDiffusionConfig, ModelForDiffusion = RobertaDiffusionConfigLM, RobertaForDiffusionLM
        elif training_args.training_data_type == "PersonaComment":
            from sdlm.data.dataset import PersonalCommentDataset_ML as PersonalCommentDataset_Seq2seq
            from sdlm.data.dataset import collate_fn_with_Comment_tokenizer as data_collator
            from sdlm.models import RobertaDiffusionConfigSeq2seq, RobertaForDiffusionSeq2seq
            ModelDiffusionConfig, ModelForDiffusion = RobertaDiffusionConfigSeq2seq, RobertaForDiffusionSeq2seq
    elif "bert" in model_args.model_name_or_path.lower():
        from sdlm.models import BertDiffusionConfig, BertForDiffusionLM
        ModelDiffusionConfig, ModelForDiffusion = BertDiffusionConfig, BertForDiffusionLM
    else:
        raise ValueError(f"{model_args.model_name_or_path} not in define model")

    config = ModelDiffusionConfig.from_pretrained(
        model_args.model_name_or_path,
        self_condition=diffusion_args.self_condition,
        self_condition_zeros_after_softmax=diffusion_args.self_condition_zeros_after_softmax,
        deepmind_conditional=diffusion_args.deepmind_conditional,
        classifier_free_simplex_inputs=diffusion_args.classifier_free_simplex_inputs,
        classifier_free_uncond_input=diffusion_args.classifier_free_uncond_input,
        self_condition_mlp_projection=diffusion_args.self_condition_mlp_projection,
        self_condition_mix_before_weights=diffusion_args.self_condition_mix_before_weights,
        self_condition_mix_logits_before_weights=diffusion_args.self_condition_mix_logits_before_weights,
        empty_token_be_mask=diffusion_args.empty_token_be_mask,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if model_args.model_name_or_path:
        model = ModelForDiffusion.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = ModelForDiffusion.from_config(config)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    if training_args.training_data_type == "Seq2seq":
        total_seq2seq_length = data_args.max_article_len + data_args.max_comment_len
    else:
        total_seq2seq_length = data_args.max_article_len

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < total_seq2seq_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {total_seq2seq_length}."
            )
            # position_ids starts from `padding_idx + 1` (padding_index=1) and we therefore requires
            # 2 more position embeddings.
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(
                total_seq2seq_length + 2,
                with_alternatation=model_args.resize_position_embeddings_alternatively,
            )
        else:
            raise ValueError(
                f"`max_source_length`+`max_target_length` is set to {total_seq2seq_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `max_source_length`+`max_target_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )


    # initialize dataset & dataloader
    train_dataset = PersonalCommentDataset_Seq2seq(data_args.train_file,
                                                   tokenizer,
                                                   data_args=data_args,)

    eval_dataset = PersonalCommentDataset_Seq2seq(data_args.validation_file,
                                                  tokenizer,
                                                  data_args=data_args,)

    test_dataset = PersonalCommentDataset_Seq2seq(data_args.test_file,
                                                  tokenizer,
                                                  data_args=data_args,)

    noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )
    inference_noise_scheduler = SimplexDDPMScheduler(
        num_train_timesteps=diffusion_args.num_inference_diffusion_steps,
        beta_schedule=diffusion_args.beta_schedule,
        simplex_value=diffusion_args.simplex_value,
        clip_sample=diffusion_args.clip_sample,
        device=training_args.device,
    )

    # Metric
    metric = evaluate.load("rouge")

    def compute_metrics(results):
        keys = ["pred_texts_from_simplex_masked", "pred_texts_from_logits_masked"]
        metrics = {}
        for key in keys:
            decoded_preds = (
                process_text(results[key])
                if not data_args.skip_special_tokens
                else results[key]
            )
            # Note that since decoded_labels is getting updated after post-process, we
            # need to compute it here for each key.
            decoded_labels = (
                process_text(results["gold_texts_masked"])
                if not data_args.skip_special_tokens
                else results["gold_texts_masked"]
            )
            decoded_preds, decoded_labels = postprocess_text_for_metric(
                "rouge", decoded_preds, decoded_labels
            )
            key_metrics = metric.compute(
                predictions=decoded_preds, references=decoded_labels, use_stemmer=True
            )
            key_metrics = {k: round(v * 100, 4) for k, v in key_metrics.items()}
            key_metrics = {f"{key}_{k}": v for k, v in key_metrics.items()}
            metrics.update(key_metrics)
        return metrics

    # Initialize our Trainer
    trainer = DiffusionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if (training_args.do_eval or training_args.do_predict)
        else None,
        noise_scheduler=noise_scheduler,
        diffusion_args=diffusion_args,
        data_args=data_args,
        inference_noise_scheduler=inference_noise_scheduler,
        wandb=wandb
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # We will load the best model here to avoid an issue when do_train is not set.
    trainer_state_path = os.path.join(model_args.model_name_or_path, "trainer_state.json")
    if (
        training_args.load_states_in_eval_from_model_path
        and not training_args.do_train
        and os.path.isfile(trainer_state_path)
    ):
        trainer.state = TrainerState.load_from_json(trainer_state_path)
        if (
            training_args.load_best_model_at_end
            and trainer.state.best_model_checkpoint is not None
        ):
            checkpoint_path = trainer.state.best_model_checkpoint
        else:
            checkpoint_path = model_args.model_name_or_path
        trainer._load_from_checkpoint(checkpoint_path)
        trainer._load_rng_state(checkpoint_path)

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = (
        data_args.num_beams
        if data_args.num_beams is not None
        else training_args.generation_num_beams
    )
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        # TODO: num_beans should be added for ours as well.
        # metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        metrics = trainer.evaluate()
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Test ***")
        metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
        max_predict_samples = (
            data_args.max_predict_samples
            if data_args.max_predict_samples is not None
            else len(test_dataset)
        )
        metrics["test_samples"] = min(max_predict_samples, len(test_dataset))
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

    # TODO: we may want to add predict part back.
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate DiffuPercom.")
    parser.add_argument("config", help="Path to a JSON experiment configuration.")
    main(parser.parse_args().config)
