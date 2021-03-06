# -*- coding: utf-8 -*-

import random, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import transformers
from config import Config
from src.bert_lr_last4layer import bert_lr_last4layer, bert_lr_last4layer_Config
from bert_lr import bert_lr, bert_lr_Config
from transformers import AdamW
from transformers import BertConfig, AlbertModel, BertModel
from util import Dataset, convert_text_to_ids, seq_padding
from utils.progressbar import ProgressBar
from utils.preprocess import Preprocess


class transformers_bert_binary_classification(object):
    def __init__(self):
        self.config = Config()
        self.device_setup()
        self.model_setup()

    def device_setup(self):
        """
        设备配置并加载BERT模型
        :return:
        """

        # 使用GPU，通过model.to(device)的方式使用
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        MODEL_PATH = self.config.get("result", "model_save_path")
        config_PATH = self.config.get("result", "config_save_path")
        vocab_PATH = self.config.get("result", "vocab_save_path")

        # 通过词典导入分词器
        self.tokenizer = transformers.BertTokenizer.from_pretrained(vocab_PATH)
        self.model_config = BertConfig.from_pretrained(config_PATH)
        self.model_type = self.config.get("BERT_path", 'model_type')

        # if self.model_type == 'Albert':
        #     self.model = AlbertModel.from_pretrained(MODEL_PATH, config=config_PATH)
        # else:
        #     self.model = BertModel.from_pretrained(MODEL_PATH, config=config_PATH)
        self.model = bert_lr(bert_lr_Config(self.config))
        self.state_dict = torch.load(MODEL_PATH)
        self.model.load_state_dict(self.state_dict)
        # self.model = bert_cnn(bert_cnn_Config(self.config))
        # self.model = bert_lr_last4layer(bert_lr_last4layer_Config(self.config))
        self.model.to(self.device)

        self.max_seq_length = self.config.get('training_rule', 'max_length')
        self.cur_time = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

    def model_setup(self):
        weight_decay = self.config.get("training_rule", "weight_decay")
        learning_rate = self.config.get("training_rule", "learning_rate")

        # 定义优化器和损失函数
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def get_data(self):
        """
        读取数据
        :return:
        """
        train_set_path = self.config.get("data_path", "trainingSet_path")
        valid_set_path = self.config.get("data_path", "valSet_path")
        batch_size = self.config.get("training_rule", "batch_size")

        # 数据读入
        # 预处理
        if self.config.get('data_preprocess_path', 'need_preprocess'):
            preprocess = Preprocess(config=self.config)
            preprocess.process()

        # 加载数据集
        train_set = Dataset(train_set_path)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        valid_set = Dataset(valid_set_path)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, valid_loader

    def eval(self):
        train_loader, valid_loader = self.get_data()

        valid_start_time = time.time()
        valid_loss, valid_acc = self.evaluate(valid_loader)
        print('valid_time: ', time.time() - valid_start_time, ' second')
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)

    def evaluate(self, data_iterator):
        pbar = ProgressBar(n_total=len(data_iterator), desc='Eval')
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        with torch.no_grad():
            for i, batch in enumerate(data_iterator):
                label = batch["label"]
                text = batch["text"]
                input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, text, self.max_seq_length)
                input_ids, input_attention_mask = seq_padding(self.tokenizer, input_ids)
                token_type_ids, _ = seq_padding(self.tokenizer, token_type_ids)
                label = label.unsqueeze(1)
                input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
                input_ids, token_type_ids, input_attention_mask, label = input_ids.to(self.device), token_type_ids.to(
                    self.device), input_attention_mask.to(self.device), label.to(
                    self.device)
                output = self.model(input_ids=input_ids, attention_mask=input_attention_mask,
                                    token_type_ids=token_type_ids, labels=label)
                # 更改了以下部分
                # y_pred_label = output[1].argmax(dim=1)
                y_pred_prob = output[1]
                y_pred_label = y_pred_prob.argmax(dim=1)
                loss = output[0]
                # loss = self.criterion(y_pred_prob.view(-1, 2), label.view(-1))
                acc = ((y_pred_label == label.view(-1)).sum()).item()
                epoch_loss += loss.item()
                epoch_acc += acc

                # 把运行结果保存起来
                # result_list.extend(list(zip(text,label.squeeze().tolist(),y_pred_label.tolist())))
                pbar(i, {'epoch_loss': epoch_loss / (i + 1), 'epoch_acc': epoch_acc / ((i + 1) * len(label))})
        return epoch_loss / len(data_iterator), epoch_acc / len(data_iterator.dataset.dataset)

    def predict(self, sentence):
        # self.model.setup()
        self.model.eval()
        # 转token后padding
        input_ids, token_type_ids = convert_text_to_ids(self.tokenizer, sentence, self.max_seq_length)
        input_ids, input_attention_mask = seq_padding(self.tokenizer, [input_ids])
        token_type_ids, _ = seq_padding(self.tokenizer, [token_type_ids])
        # 需要 LongTensor
        input_ids, token_type_ids = input_ids.long(), token_type_ids.long()
        # 梯度清零
        self.optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids, input_attention_mask = input_ids.to(self.device), token_type_ids.to(
            self.device), input_attention_mask.to(self.device)
        output = self.model(input_ids=input_ids, attention_mask=input_attention_mask, token_type_ids=token_type_ids)
        # y_pred_prob:各个类别的概率
        y_pred_prob = output[0]
        # 取概率最大的标签
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 将torch.tensor转换回int形式
        return y_pred_label.item()


if __name__ == '__main__':
    classifier = transformers_bert_binary_classification()
    classifier.eval()
    print(classifier.predict("交通不好"))  # 0
    print(classifier.predict("这个书不推荐看"))  # 0
    print(classifier.predict("交通挺好的"))  # 1
    print(classifier.predict("这个衣服很漂亮"))  # 1
    print(classifier.predict("酒店的服务人员服务周到，细致，给人一种宾至如归的感觉"))  # 1
