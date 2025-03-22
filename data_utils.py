import logging
import os
import random
import time
import numpy as np

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, XLMRobertaTokenizerFast

from seq_utils import ot2bio, ot2bieos

# random.seed(42)
logger = logging.getLogger(__name__)


def get_tag_list(task, tagging_schema):
    task = task.lower()
    tagging_schema = tagging_schema.lower()
    if task == 'absa':
        if tagging_schema == 'ot':
            return ['O', 'T-POS', 'T-NEG', 'T-NEU']
        elif tagging_schema == 'bio':
            return ['O', 'B-POS', 'I-POS', 'B-NEG', 'I-NEG', 'B-NEU', 'I-NEU']
        elif tagging_schema == 'bieos':
            return ['O', 'B-POS', 'I-POS', 'E-POS', 'S-POS',
                    'B-NEG', 'I-NEG', 'E-NEG', 'S-NEG',
                    'B-NEU', 'I-NEU', 'E-NEU', 'S-NEU']
        else:
            raise Exception("Invalid tagging schema: {}".format(tagging_schema))
    elif task == 'ate':
        if tagging_schema == 'ot':
            return ['O', 'T']
        elif tagging_schema == 'bio':
            return ['O', 'B', 'I']
        elif tagging_schema == 'bieos':
            return ['O', 'B', 'I', 'E', 'S']
        else:
            raise Exception("Invalid tagging schema: {}".format(tagging_schema))
    else:
        raise Exception("Invalid task name: {}".format(task))


def get_tag_vocab(task, tagging_schema, label_path):
    if label_path == '':
        tag_list = get_tag_list(task, tagging_schema)
    else:
        tag_list = []
        with open(label_path) as f:
            for line in f:
                if line.strip():
                    tag_list.append(line.strip())
    tag2idx = {tag: idx for idx, tag in enumerate(tag_list)}
    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
    return tag_list, tag2idx, idx2tag


def read_examples_from_file(file_path, task, tagging_schema, ratio=1.0):
    sents, labels = [], []
    logger.info(f"Read data from file {file_path}")
    with open(file_path, 'r', encoding='UTF-8') as fp:
        words, tags = [], []
        for line in fp:
            line = line.strip()
            if line != '':
                word, tag, _ = line.split('\t')
                words.append(word)
                tags.append(tag)
            else:
                if task.lower() == 'ate':
                    tags = [tag[0] for tag in tags]
                if tagging_schema.lower() == 'bieos':
                    tags = ot2bieos(tags, task)
                elif tagging_schema.lower() == 'bio':
                    tags = ot2bio(tags, task)
                else:
                    pass
                sents.append(words)
                labels.append(tags)
                words, tags = [], []

    if ratio < 1.0:
        n_samples = int(len(sents) * ratio)
        sample_indices = random.sample(range(len(sents)), n_samples)
        sents = [sents[i] for i in sample_indices]
        labels = [labels[i] for i in sample_indices]

    print(f"Total examples = {len(sents)}")
    return sents, labels


def read_examples_from_multiple_file(file_path_list, task, tagging_schema, exp_type, ratio):
    all_sents, all_labels = [], []
    for file_path in file_path_list:
        lang = file_path.split('-')[1]
        sents, labels = read_examples_from_file(file_path, task, tagging_schema)
        all_sents.extend(sents)
        all_labels.extend(labels)
    assert len(all_sents) == len(all_labels)
    print(f"** Total examples (involving multiple files) = {len(all_sents)}")
    return list(all_sents), list(all_labels)


def fix_space_issue(encodings, tokenizer):
    input_ids = encodings.input_ids
    offset_mappings = encodings.offset_mapping

    num_sents = len(input_ids)
    for i in range(num_sents):
        num_tokens = len(input_ids[i])
        for j in range(num_tokens):
            if input_ids[i][j] == 6 and offset_mappings[i][j] == (0, 1):
                offset_mappings[i][j] = (0, 0)
            if offset_mappings[i][j] == (0, 0) and input_ids[i][j] not in [0, 1, 2, 6]:
                print(tokenizer.convert_ids_to_tokens(input_ids[i][j]), "(will be kept)")
                offset_mappings[i][j] = (0, 1)

    encodings.offset_mapping = offset_mappings
    return encodings


def encode_tags(tags, tag2id, offset_mapping):
    labels = [[tag2id[tag] for tag in doc] for doc in tags]
    encoded_labels = []
    i = 0
    for doc_labels, doc_offset in zip(labels, offset_mapping):
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
        arr_offset = np.array(doc_offset)
        mask = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        if len(doc_labels) > 300:
            num_positive_positions = sum([1 for pos in mask if pos])
            doc_enc_labels[mask] = doc_labels[:num_positive_positions]
        else:
            doc_enc_labels[mask] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        i += 1
    return encoded_labels


class XABSADataset(Dataset):
    def __init__(self, encodings, labels,languages=None):
        self.encodings = encodings
        self.labels = labels
        self.languages = languages
            # languages if languages is not None else ['unknown'] * len(labels)#v1

        # print(f"encodings:{len(self.encodings['input_ids']),labels:{len(self.labels)},languages:{len(self.languages)}}")


    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        item['language'] = self.languages[idx]  #v1
        return item

    def __len__(self):
        return len(self.labels)


def compute_language_weights(args, tokenizer, model):
    """
    计算语言相似性权重，仅使用模型的嵌入层
    """
    import torch.nn.functional as F

    lang_list = ['en', 'fr', 'es', 'nl', 'ru']
    lang_embeddings = {}

    for lang in lang_list:
        file_path = f"{args.data_dir}/rest/gold-{lang}-train.txt"
        if not os.path.exists(file_path):
            logger.warning(f"Language file {file_path} not found, skipping {lang}")
            continue

        texts, _ = read_examples_from_file(file_path, args.task, args.tagging_schema)
        word_freq = {}
        for sent in texts:
            for word in sent:
                word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:100]
        top_words = [w[0] for w in top_words]

        encodings = tokenizer(top_words, is_split_into_words=False, return_tensors='pt', padding=True, truncation=True,
                              max_length=args.max_seq_length)
        encodings = {k: v.to(args.device) for k, v in encodings.items()}

        # 只使用嵌入层，避免调用完整 forward
        with torch.no_grad():
            if args.tfm_type == 'xlmr':
                embeddings = model.roberta.embeddings(encodings['input_ids'])  # (100, seq_len, hidden_size)
            else:  # bert or mbert
                embeddings = model.bert.embeddings(encodings['input_ids'])
            lang_embeddings[lang] = embeddings.mean(dim=1).mean(dim=0)  # (hidden_size,)

    weights = torch.zeros(len(lang_list), len(lang_list))
    for i, lang_i in enumerate(lang_list):
        if lang_i not in lang_embeddings:
            continue
        for j, lang_j in enumerate(lang_list):
            if lang_j not in lang_embeddings:
                continue
            cos_sim = F.cosine_similarity(lang_embeddings[lang_i], lang_embeddings[lang_j], dim=0)
            weights[i, j] = (cos_sim + 1) / 2

    weights = weights / weights.sum()
    logger.info(f"Computed language weights: \n{weights}")
    return weights.to(args.device)


def build_or_load_dataset(args, tokenizer, mode='train'):
    _, tag2idx, _ = get_tag_vocab(task=args.task, tagging_schema=args.tagging_schema,
                                  label_path=args.label_path)
    exp_type = args.exp_type

    if mode == 'dev':
        file_name_or_list = f'gold-{args.src_lang}-dev.txt'
    elif mode == 'test':
        file_name_or_list = f"gold-{args.tgt_lang}-test.txt"
    elif mode == 'unlabeled':
        file_name_or_list = f"gold-{args.tgt_lang}-train.txt"
    elif mode == 'unlabeled_mtl':
        file_name_or_list = [f"gold-{l}-train.txt" for l in ['fr', 'es', 'nl', 'ru']]
    elif mode == 'train':
        if exp_type == 'supervised':
            assert args.src_lang == args.tgt_lang
            file_name_or_list = f"gold-{args.src_lang}-train.txt"
        elif exp_type == 'smt':
            file_name_or_list = f"{exp_type}-{args.tgt_lang}-train.txt"
        elif exp_type.startswith('mtl'):
            file_name_or_list = ['gold-en-train.txt'] + [f'smt-{l}-train.txt' for l in ['fr', 'es', 'nl', 'ru']]
        elif exp_type == 'acs':
            file_name_or_list = [f"gold-{args.src_lang}-train.txt",
                                 f"cs_{args.src_lang}-{args.tgt_lang}-train.txt",
                                 f"cs_{args.tgt_lang}-{args.src_lang}-train.txt",
                                 f"smt-{args.tgt_lang}-train.txt"]
        elif exp_type == 'acs_mtl':
            lang_list = ['fr', 'es', 'nl', 'ru']
            file_name_or_list = ["gold-en-train.txt"] + \
                                [f'smt-{l}-train.txt' for l in lang_list] + \
                                [f'cs_en-{l}-train.txt' for l in lang_list] + \
                                [f'cs_{l}-en-train.txt' for l in lang_list]
        else:
            raise Exception(f"Invalid exp_type `{exp_type}`")
    else:
        raise Exception(f"Invalid mode `{mode}`")

    logger.info(f"We will read file from {file_name_or_list} as {mode.upper()} data")

    top_data_dir = f"{args.data_dir}/rest"
    if isinstance(file_name_or_list, str):
        file_path = f"{top_data_dir}/{file_name_or_list}"
        cached_features_file = "{0}/cached-{1}-{2}-{3}-{4}".format(
            top_data_dir, args.task, args.tfm_type, exp_type, file_name_or_list[:-4])
        lang_list = [file_path.split('-')[1].split('.')[0]]
    elif isinstance(file_name_or_list, list):
        file_path = [f"{top_data_dir}/{f}" for f in file_name_or_list]
        included_sets = 'train_train' if exp_type.startswith('bilingual') else 'xxx'
        cached_features_file = "{0}/cached-{1}-{2}-{3}-mixed-{4}-{5}".format(
            top_data_dir, args.task, args.tfm_type, exp_type, args.tgt_lang, included_sets)
        lang_list = [f.split('-')[1].split('.')[0] for f in file_name_or_list]

    if os.path.exists(cached_features_file) and not args.ignore_cached_data:
        logger.info(f"Find cached_features_file: {cached_features_file}")
        encodings, encoded_labels = torch.load(cached_features_file)
        languages = ['unknown'] * len(encoded_labels)
    else:
        logger.info(f"Didn't find / Ignore cached_features_file: {cached_features_file}, create and save...")
        if isinstance(file_name_or_list, str):
            texts, tags = read_examples_from_file(file_path, args.task, args.tagging_schema, ratio=args.data_select)
            languages = [lang_list[0]*len(texts)]
        elif isinstance(file_name_or_list, list):
            texts,tags = [] ,[]
            languages = []
            for i,fp in enumerate(file_path):
                sub_texts,sub_tags = read_examples_from_file(fp,args.task,args.tagging_schema,ratio=args.data_select)
                texts.extend(sub_texts)
                tags.extend(sub_tags)
                languages.extend([lang_list[i]*len(sub_texts)])

            #texts, tags = read_examples_from_multiple_file(file_path, args.task, args.tagging_schema, exp_type,
            #                                                ratio=0.5)

        assert isinstance(tokenizer, BertTokenizerFast) or isinstance(tokenizer, XLMRobertaTokenizerFast)
        encodings = tokenizer(texts, is_split_into_words=True, return_offsets_mapping=True, truncation=True,
                              padding='max_length', max_length=args.max_seq_length)

        if args.tfm_type == 'xlmr':
            encodings = fix_space_issue(encodings, tokenizer)
        encoded_labels = encode_tags(tags, tag2idx, encodings.offset_mapping)
        encodings.pop("offset_mapping")

        # print(f'languages:{len(languages)}')
        # print(languages)
        # raise Exception('TTTTT')
        ll = []
        for i in languages:
            for j in range(0, len(i), 2):
                seg = i[j:j + 2]
                ll.append(seg)

        if args.local_rank in [-1, 0]:
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save((encodings, encoded_labels,ll), cached_features_file)

    dataset = XABSADataset(encodings, encoded_labels,ll)
    return dataset


def build_contrastive_dataset(args, tokenizer):
    logger.info("Building contrastive dataset from external files...")
    languages = ['en', 'fr', 'es', 'nl', 'ru']
    data_files = {
        lang: f"{args.contrastive_data_dir}/contrastive_{lang}.txt"
        for lang in languages
    }

    all_texts = {}
    for lang, file_path in data_files.items():
        with open(file_path, 'r', encoding='utf-8') as fp:
            texts = [line.strip().split() for line in fp if line.strip()]  # 分词
            all_texts[lang] = texts

    min_samples = min(len(texts) for texts in all_texts.values())
    logger.info(f"Aligning contrastive data to minimum sample size: {min_samples}")
    for lang in languages:
        all_texts[lang] = all_texts[lang][:min_samples]

    encodings = {}
    for lang in languages:
        encodings[lang] = tokenizer(all_texts[lang], is_split_into_words=True, return_offsets_mapping=True,
                                    truncation=True, padding='max_length', max_length=args.max_seq_length)
        if args.tfm_type == 'xlmr':
            encodings[lang] = fix_space_issue(encodings[lang], tokenizer)
        encodings[lang].pop("offset_mapping")

    dataset = XABSADatasetContrastive(encodings, languages)
    return dataset


class XABSADatasetContrastive(Dataset):
    def __init__(self, encodings, languages):
        self.encodings = encodings
        self.languages = languages
        self.num_samples = len(encodings[languages[0]]['input_ids'])

    def __getitem__(self, idx):
        item = {f"{lang}_{key}": torch.tensor(value[idx])
                for lang in self.languages
                for key, value in self.encodings[lang].items()}
        item['languages'] = self.languages
        return item

    def __len__(self):
        return self.num_samples

def write_results_to_log(log_file_path, best_test_result, args, dev_results, test_results, global_steps):
    local_time = time.asctime(time.localtime(time.time()))
    exp_settings = "Exp setting: {0} for {1} ({7}) | {6:.4f} | {2} -> {3} in {4} setting.\nModel is saved in {5}".format(
        args.tfm_type, 'XABSA', args.src_lang, args.tgt_lang, args.exp_type,
        args.saved_model_dir, best_test_result, args.tagging_schema
    )
    train_settings = "Train setting: bs={0}, lr={1}, total_steps={2} (Start eval from {3})".format(
        args.per_gpu_train_batch_size, args.learning_rate, args.max_steps, args.train_begin_saving_step
    )
    results_str = "\n* Results *:  Dev  /  Test  \n"
    metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
    for gstep in global_steps:
        results_str += f"Step-{gstep}:\n"
        for name in metric_names:
            name_step = f'{name}_{gstep}'
            results_str += f"{name:<8}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}"
            results_str += ' ' * 5
        results_str += '\n'

    log_str = f"{local_time}\n{exp_settings}\n{train_settings}\n{results_str}\n\n"
    # with open('log.txt', "a+") as f:
    with open('log.txt', "a+") as f:
        f.write(log_str)
    with open(log_file_path, "a+") as f:
        f.write(log_str)