# -*- coding: utf-8 -*-
"""
手动实现transformer.models.bert.BertForSequenceClassification()函数
可以方便理解其使用，对bert+其他模型进行构建

"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import BertModel,BertTokenizer,BertConfig,AlbertForMaskedLM,AlbertModel
import torch.nn.functional as F


class bert_lr_Config(nn.Module):
    def __init__(self,config):
        self.config = config
        self.model_type = config.get("BERT_path",'model_type')

        continue_train = self.config.get('training_rule', 'continue_train')
        if not continue_train:
            self.bert_path = self.config.get("BERT_path", 'file_path')
            self.config_path = self.config.get("BERT_path", 'config_path')
        else:
            self.bert_path = self.config.get("result", 'model_save_path')
            self.config_path = self.config.get("result", 'config_save_path')

        # self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = self.config.get("training_rule", 'hidden_size')
        self.num_labels = self.config.get("training_rule", 'num_labels')
        self.dropout_bertout = self.config.get("training_rule", 'hidden_dropout_prob')
        self.mytrainedmodel = self.config.get("result", 'model_save_path')
        """
        current loss: 0.35029613971710205 	 current acc: 0.90625
        current loss: 0.12146440659422632 	 current acc: 0.9614427860696517
        current loss: 0.1080296240059515 	 current acc: 0.96571072319202
        train loss:  0.1104624584773149 	 train acc: 0.964012114733654
        valid loss:  0.15432240582551016 	 valid acc: 0.9445760303173851
        """

class bert_lr(nn.Module):

    def __init__(self,config):
        super(bert_lr, self).__init__()
        self.model_config = BertConfig.from_pretrained(config.config_path, num_labels=config.num_labels,
                                                       hidden_dropout_prob=config.dropout_bertout)
        if config.model_type == 'Albert':
            self.bert = AlbertModel.from_pretrained(config.bert_path, config=config.config_path)
        else:
            self.bert = BertModel.from_pretrained(config.bert_path,config = config.config_path)
        self.dropout_bertout = nn.Dropout(config.dropout_bertout)
        self.num_labels = config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1] # [1]是pooled的结果
        # print(pooled_output.shape)
        pooled_output = self.dropout_bertout(pooled_output)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss,logits


if __name__ == "__main__":
    model = bert_lr(config=bert_lr_Config())
    print(model)