

import argparse
import os
import logging
import random
import pickle
import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertQPENTagger, XLMRQPENTagger
from seq_utils import compute_metrics_absa
from data_utils import build_or_load_dataset, get_tag_vocab, write_results_to_log, compute_language_weights,build_contrastive_dataset

from time import strftime, localtime
import  matplotlib.pyplot as plt
from sklearn.manifold import TSNE


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, BertQPENTagger, BertTokenizerFast),
    'mbert': (BertConfig, BertQPENTagger, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRQPENTagger, XLMRobertaTokenizerFast)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='absa', type=str, help="absa or ate ")
    parser.add_argument("--tfm_type", default='xlmr', type=str, required=True,
                        help="The base transformer, selected from: [bert, mbert, xlmr]")
    parser.add_argument("--model_name_or_path", default='xlm-roberta-base', type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--exp_type", default='acs', type=str,
                        help="Experiment type, selected from: [supervised, zero_shot, ...]")
    parser.add_argument("--data_dir", default='./data/', type=str, help="Base data dir")
    parser.add_argument("--src_lang", default='en', type=str, required=True, help="source language")
    parser.add_argument("--tgt_lang", default='fr', type=str, required=True, help="target language, sp = es, du = nl")
    parser.add_argument("--data_select", default=1.0, type=float, help="ratio of the selected data to train")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument("--ignore_cached_data", default=True, action='store_true')
    parser.add_argument("--train_data_sampler", type=str, default='', help='random or another str ')
    parser.add_argument("--do_lower_case", action='store_true')
    parser.add_argument("--config_name", default="", type=str)
    parser.add_argument("--tokenizer_name", default="", type=str)
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=200, type=int)
    parser.add_argument("--do_train", action='store_true', default=True)
    parser.add_argument("--trained_teacher_paths", type=str, default='./outputs/base/checkpoint')
    parser.add_argument("--do_eval", action='store_true', default=True)
    parser.add_argument("--do_contrastive_pretrain", action='store_true', default=False,
                        help="Whether to perform contrastive pre-training")
    parser.add_argument("--contrastive_data_dir", default='./data/contrastive/', type=str,
                        help="Directory containing contrastive learning data")
    parser.add_argument("--contrastive_epochs", default=3, type=int,
                        help="Number of epochs for contrastive pre-training")
    parser.add_argument("--contrastive_lr", default=2e-5, type=float,
                        help="Learning rate for contrastive pre-training")
    parser.add_argument("--contrastive_tau", default=0.1, type=float,
                        help="Temperature parameter for contrastive loss")
    parser.add_argument("--eval_begin_end", default="1500-2000", type=str)
    parser.add_argument("--evaluate_during_training", action='store_true')
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int)
    parser.add_argument("--per_gpu_eval_batch_size", default=12, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--freeze_bottom_layer", default=-1, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=2.8e-5, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--num_train_epochs", default=20.0, type=float)
    parser.add_argument("--max_steps", default=2000, type=int)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument("--train_begin_saving_step", type=int, default=1500)
    parser.add_argument("--train_begin_saving_epoch", type=int, default=10)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument("--eval_all_checkpoints", action='store_true')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument('--overwrite_output_dir', default=True, action='store_true')
    parser.add_argument('--overwrite_cache', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--MASTER_ADDR', type=str, default='localhost')
    parser.add_argument('--MASTER_PORT', type=str, default='23455')
    parser.add_argument("--local_rank", type=int, default=-1)


    args = parser.parse_args()
    output_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{args.tgt_lang}-{args.exp_type}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir
    args.saved_model_dir = output_dir  # 添加 saved_model_dir
    return args


def compute_contrastive_loss(features, languages, batch_size, tau=0.1):
    num_languages = len(languages)
    features = features.view(batch_size, num_languages, -1)
    total_loss = 0.0
    for i in range(num_languages):
        src_features = features[:, i]
        pos_features = features[:, list(range(num_languages)) != i]
        pos_sim = torch.cosine_similarity(src_features.unsqueeze(1), pos_features, dim=-1) / tau
        neg_sim = torch.cosine_similarity(src_features.unsqueeze(1), features[:, i].unsqueeze(0), dim=-1) / tau
        neg_sim = neg_sim.masked_fill(torch.eye(batch_size, device=features.device).bool(), float('-inf'))
        pos_exp = torch.exp(pos_sim).sum(dim=-1)
        neg_exp = torch.exp(neg_sim).sum(dim=-1)
        loss = -torch.log(pos_exp / (pos_exp + neg_exp + 1e-10)).mean()
        total_loss += loss

    return total_loss / num_languages


def contrastive_pretrain(args, train_dataset, model, tokenizer):
    logger.info("***** Running contrastive pre-training with external data *****")
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)

    optimizer = AdamW(model.parameters(), lr=args.contrastive_lr, eps=args.adam_epsilon)
    num_steps = len(train_dataloader) * args.contrastive_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_steps)

    global_step = 0
    train_loss = 0.0
    model.zero_grad()

    for epoch in trange(args.contrastive_epochs, desc="Contrastive Pretrain Epoch"):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            all_features = []
            for lang in batch['languages']:
                inputs = {
                    'input_ids': batch[f'{lang}_input_ids'].to(args.device),
                    'attention_mask': batch[f'{lang}_attention_mask'].to(args.device),
                    'token_type_ids': batch[f'{lang}_token_type_ids'].to(
                        args.device) if args.tfm_type != 'xlmr' else None,
                    'contrastive_mode': True
                }
                features = model(**inputs)
                all_features.append(features)

            all_features = torch.cat(all_features, dim=0)
            loss = compute_contrastive_loss(all_features, batch['languages'],
                                            batch_size=args.per_gpu_train_batch_size, tau=args.contrastive_tau)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            train_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1
                if args.local_rank in [-1, 0] and global_step % args.logging_steps == 0:
                    logger.info(f"Contrastive Pretrain Step {global_step}, Loss: {loss.item():.4f}")

            if global_step >= num_steps:
                break

        if global_step >= num_steps:
            break
    logger.info(f"Contrastive pre-training completed. Total steps: {global_step}, Avg loss: {train_loss / global_step:.4f}")
    return model


def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info(" The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info("   Freeze: %s", n)
        logger.info(" The parameters to be fine-tuned are:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info("   Fine-tune: %s", n)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                print(n)
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]
    return outputs


def train(args, train_dataset, model, tokenizer):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if
                                    layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps * (
        torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)

    # pretrain_features = []
    # softmax_features = []
    # all_languages = []


    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                'labels': batch['labels'].to(args.device),
                'lang_weights': args.lang_weights if hasattr(args, 'lang_weights') else None
                # 'extract_features': True
            }
            #


            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1,0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break

        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")


    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step


def evaluate(args, eval_dataset, model, idx2tag, mode, step=None):
    eval_output_dir = args.output_dir
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}

    # pretrain_features = []
    # softmax_features = []
    # all_languages = []


    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {
                'input_ids': batch['input_ids'].to(args.device),
                'attention_mask': batch['attention_mask'].to(args.device),
                'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                'labels': batch['labels'].to(args.device),
                'lang_weights': args.lang_weights if hasattr(args, 'lang_weights') else None
                # 'extract_features': True
            }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        num_eval_steps += 1
        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)


    eval_loss = eval_loss / num_eval_steps
    pred_labels = np.argmax(preds, axis=-1)
    result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write = {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")

    return results

def visualize_features(features, languages, output_dir, title, filename):
    target_langs = ['fr', 'es', 'nl', 'ru','en']
    lang_features = {lang: [] for lang in target_langs}
    for feature, lang in zip(features, languages):
        if lang in target_langs:
            lang_features[lang].append(feature.cpu().numpy())

    all_features = []
    all_labels = []
    for lang in target_langs:
        if lang_features[lang]:
            all_features.extend(lang_features[lang])
            all_labels.extend([lang] * len(lang_features[lang]))

    if not all_features:
        logger.info(f"No features found for visualization in {title}.")
        return

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    all_features = all_features.mean(axis=1)

    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8))
    colors = {'fr': 'blue', 'es': 'green', 'nl': 'orange', 'ru': 'red'}
    for lang in target_langs:
        mask = all_labels == lang
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], c=colors[lang], label=lang.upper(), alpha=0.5)
    plt.legend()
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    logger.info(f"Feature visualization saved to {output_dir}/{filename}")



def main():
    args = init_args()
    print("\n", "=" * 30, f"NEW EXP ({args.src_lang} -> {args.tgt_lang} for {args.exp_type}", "=" * 30, "\n")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty.")

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}")
    logger.info(f"Distributed training: {bool(args.local_rank != -1)}, 16-bits training: False")

    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform QABSA task with label list being {tag_list} (n_labels={num_tags})")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    if args.tfm_type == 'mbert':
        lr, batch_size = 5e-5, 16
    elif args.tfm_type == 'xlmr':
        lr, batch_size = 4e-5, 25
    # args.learning_rate = lr
    # args.per_gpu_train_batch_size = batch_size
    logger.info(f"We hard-coded set lr={args.learning_rate} and bs={args.per_gpu_train_batch_size}")

    if args.do_train:
        logger.info("\n\n***** Prepare to conduct training  *****\n")
        args.tfm_type = args.tfm_type.lower()
        logger.info(f"Load pre-trained {args.tfm_type} model from `{args.model_name_or_path}`")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        if os.path.exists(args.model_name_or_path):
            config = config_class.from_pretrained(args.model_name_or_path,num_labels=num_tags,id2label=idx2tag,label2id=tag2idx)
            tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path,do_lower_case=args.do_lower_case)
            model = model_class.from_pretrained(args.model_name_or_path, config=config)
            logger.info(f"test1111")
        else:
            config = config_class.from_pretrained(
                args.config_name if args.config_name else args.model_name_or_path,
                num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
            )
            tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case
            )
            model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)

        # 在加载数据集之前计算语言权重
        args.model = model  # 将模型赋值给 args.model
        if args.exp_type in ['acs', 'acs_mtl', 'mtl']:
            args.lang_weights = compute_language_weights(args, tokenizer, model)

        if args.do_contrastive_pretrain:
            logger.info("Prepare contrastive training examples from external data...")
            contrastive_dataset = build_contrastive_dataset(args, tokenizer)
            model = contrastive_pretrain(args, contrastive_dataset, model, tokenizer)

        print()
        logger.info("Prepare training examples...")
        train_dataset = build_or_load_dataset(args, tokenizer, mode='train')

        _, _ = train(args, train_dataset, model, tokenizer)

        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

    if args.do_eval:
        exp_type = args.exp_type
        logger.info("\n\n***** Prepare to conduct evaluation *****\n")
        logger.info(f"We are evaluating for *{args.tgt_lang}* under *{args.exp_type}* setting...")

        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_global_step = -999999.0, None, None
        all_checkpoints, global_steps = [], []

        if 'mtl1' in exp_type:
            one_tgt_lang = 'fr'
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{one_tgt_lang}-{exp_type}"
        elif exp_type == 'zero_shot':
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{args.src_lang}-supervised"
            if not os.path.exists(saved_model_dir):
                raise Exception("No trained models can be found!")
        else:
            saved_model_dir = args.output_dir

        args.saved_model_dir = saved_model_dir

        for f in os.listdir(saved_model_dir):
            sub_dir = os.path.join(saved_model_dir, f)
            if os.path.isdir(sub_dir):
                all_checkpoints.append(sub_dir)
        logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(all_checkpoints[0])
        tokenizer = tokenizer_class.from_pretrained(all_checkpoints[0])
        logger.info("Load DEV dataset...")
        dev_dataset = build_or_load_dataset(args, tokenizer, mode='dev')
        logger.info("Load TEST dataset...")
        test_dataset = build_or_load_dataset(args, tokenizer, mode='test')

        dir_path = './txt/{}_{}_{}_{}'.format(args.tfm_type, args.exp_type, args.tgt_lang,
                                              strftime("%Y-%m-%d_%H:%M:%S", localtime()))
        os.makedirs(dir_path)

        for checkpoint in all_checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoint) > 1 else ""
            eval_begin, eval_end = args.eval_begin_end.split('-')
            if int(eval_begin) <= int(global_step) < int(eval_end):
                global_steps.append(global_step)
                logger.info(f"\nLoad the trained model from {checkpoint}...")
                model = model_class.from_pretrained(checkpoint, config=config)
                model.to(args.device)
                dev_result = evaluate(args, dev_dataset, model, idx2tag, mode='dev')
                metrics = 'micro_f1'
                if dev_result[metrics] > best_f1:
                    best_f1 = dev_result[metrics]
                    best_checkpoint = checkpoint
                    best_global_step = global_step

                dev_result = dict((k + '_{}'.format(global_step), v) for k, v in dev_result.items())
                dev_results.update(dev_result)
                test_result = evaluate(args, test_dataset, model, idx2tag, mode='test', step=global_step)
                test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
                test_results.update(test_result)

        logger.info(f"\n\nThe best checkpoint is {best_checkpoint}")
        best_step_metric = f"{metrics}_{best_global_step}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")
        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
        for gstep in global_steps:
            print(f"Step-{gstep}:")
            for name in metric_names:
                name_step = f'{name}_{gstep}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.tfm_type}-{args.exp_type}-{args.tgt_lang}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results,
                             global_steps)


if __name__ == '__main__':
    main()