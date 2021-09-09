import torch

import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel,AutoModel
import torch.nn.functional as F
import numpy as np
import math
import os



PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'scibert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class ddi_Bert(BertPreTrainedModel):
    def __init__(self, config, args):
        super(ddi_Bert, self).__init__(config)

        self.args = args
        self.num_labels = config.num_labels
        self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)

        if self.args.model == "only_bert":
            self.label_classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)


    def forward(self,guid, input_ids, attention_mask, token_type_ids,
                labels):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )

        sequence_output = outputs[0]

        # only Bert model used
        if self.args.model == "only_bert":
            logits = self.label_classifier(sequence_output)
            outputs = (logits,) + outputs[2:]


        # Softmax

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # 交叉熵损失函数
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # if labels is not None:
        #     if self.num_labels == 1:
        #         loss_fct = nn.MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = nn.CrossEntropyLoss()
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #
        #     outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






