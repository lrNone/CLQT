import torch
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from transformers import PreTrainedModel, RobertaModel, RobertaConfig

import torch.nn.functional as F
from torch import nn
from Quantum import PositionEmbedding, ComplexMultiply, QOuter, QMixture, QMeasurement,DynamicEntanglement


class L2Norm(nn.Module):
    def __init__(self, dim=1, keep_dims=True, eps=1e-10):
        super(L2Norm, self).__init__()
        self.dim = dim
        self.keepdim = keep_dims
        self.eps = eps

    def forward(self, inputs):
        output = torch.sqrt(self.eps + torch.sum(inputs**2, dim=self.dim, keepdim=self.keepdim))
        return output


class BertQPENTagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertQPENTagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        penultimate_hidden_size = bert_config.hidden_size

        # 量子模块参数
        self.seq_len = 200
        self.dim = 50
        self.emb_dim = 100
        self.liner = nn.Linear(self.seq_len, self.dim)
        self.norm = L2Norm(dim=-1)
        self.projections = nn.Linear(penultimate_hidden_size, self.emb_dim)
        self.phase_embeddings = PositionEmbedding(self.emb_dim, input_dim=1)
        self.multiply = ComplexMultiply()
        self.mixture = QMixture()
        self.outer = QOuter()
        self.measurement = QMeasurement(self.emb_dim)

        # 新增动态纠缠模块
        self.num_languages = 5  # EN, FR, SP, DU, RU
        self.entanglement = DynamicEntanglement(self.emb_dim, self.num_languages)

        # 分类器：增加纠缠表示的维度
        self.classifier = nn.Linear(penultimate_hidden_size + self.dim + self.emb_dim * 2, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None, lang_reps=None, lang_weights=None,extract_features=False,contrastive_mode=False):
        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)
        tagger_input = outputs[0]  # (bsz, seq_len, hidden_size)

        if contrastive_mode:
            cls_reps = tagger_input[:, 0, :]  # (bsz, hidden_size)
            return cls_reps

        # 量子投影模块
        utterance_reps = nn.ReLU()(self.projections(tagger_input))  # (bsz, seq_len, emb_dim)
        phases = self.phase_embeddings(attention_mask)
        amplitudes = F.normalize(utterance_reps, dim=-1)
        unimodal_pure = self.multiply([phases, amplitudes])
        unimodal_matrices = self.outer(unimodal_pure)
        weights = self.norm(utterance_reps)
        weights = F.softmax(weights, dim=-1)
        in_states = self.mixture([[unimodal_matrices], weights])
        output = [self.measurement(_h) for _h in in_states]
        output = torch.stack(output, dim=-2)  # (bsz, seq_len, emb_dim)

        # 动态纠缠模块
        if lang_reps is not None and lang_weights is not None:
            entangled_reps = self.entanglement(lang_reps, lang_weights)  # (bsz, seq_len, emb_dim)
        else:
            entangled_reps = torch.zeros_like(output)  # 默认零向量

        # 额外变换
        tagger_input_transformed = tagger_input @ torch.transpose(tagger_input, -1, -2)
        tagger_input_transformed = nn.ReLU()(self.liner(tagger_input_transformed))

        # 拼接所有表示
        tagger_input = torch.cat([tagger_input, tagger_input_transformed, output, entangled_reps], dim=-1)
        tagger_input = self.bert_dropout(tagger_input)

        if extract_features:
            return outputs[0],tagger_input


        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs



class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple 
    interface for downloading and loading pretrained models.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class XLMRQPENTagger(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.seq_len = 200
        self.dim = 50
        self.emb_dim = 100
        self.liner = nn.Linear(self.seq_len, self.dim)
        self.norm = L2Norm(dim=-1)
        self.projections = nn.Linear(config.hidden_size, self.emb_dim)
        self.phase_embeddings = PositionEmbedding(self.emb_dim, input_dim=1)
        self.multiply = ComplexMultiply()
        self.mixture = QMixture()
        self.outer = QOuter()
        self.measurement = QMeasurement(self.emb_dim)

        self.num_languages = 5
        self.entanglement = DynamicEntanglement(self.emb_dim, self.num_languages)

        self.classifier = nn.Linear(config.hidden_size + self.dim + self.emb_dim * 2, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None, lang_reps=None, lang_weights=None,extract_features=False,contrastive_mode=False):
        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                               attention_mask=attention_mask, head_mask=head_mask)
        tagger_input = outputs[0]

        if contrastive_mode:
            cls_reps = tagger_input[:, 0, :]  # (bsz, hidden_size)
            return cls_reps

        utterance_reps = nn.ReLU()(self.projections(tagger_input))
        phases = self.phase_embeddings(attention_mask)
        amplitudes = F.normalize(utterance_reps, dim=-1)
        unimodal_pure = self.multiply([phases, amplitudes])
        unimodal_matrices = self.outer(unimodal_pure)
        weights = self.norm(utterance_reps)
        weights = F.softmax(weights, dim=-1)
        in_states = self.mixture([[unimodal_matrices], weights])
        output = [self.measurement(_h) for _h in in_states]
        output = torch.stack(output, dim=-2)

        if lang_reps is not None and lang_weights is not None:
            entangled_reps = self.entanglement(lang_reps, lang_weights)
        else:
            entangled_reps = torch.zeros_like(output)

        tagger_input_transformed = tagger_input @ torch.transpose(tagger_input, -1, -2)
        tagger_input_transformed = nn.ReLU()(self.liner(tagger_input_transformed))
        tagger_input = torch.cat([tagger_input, tagger_input_transformed, output, entangled_reps], dim=-1)
        tagger_input = self.dropout(tagger_input)

        if extract_features:
            return outputs[0],tagger_input

        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs