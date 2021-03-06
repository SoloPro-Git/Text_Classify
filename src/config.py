# -*- coding: utf-8 -*-

"""
配置模型、路径、与训练相关参数
"""

class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_preprocess_path": {
                "data_path": "../../data/LevelOneIntent/测试数据.xlsx",
                "train_path": "../../data/LevelOneIntent/train.data",
                "test_path": "../../data/LevelOneIntent/test.data",
                "need_preprocess":False
            },
            "data_path": {
                "trainingSet_path": "../data/sentiment/sentiment.train.data",
                "valSet_path": "../data/sentiment/sentiment.valid.data",
                "testingSet_path": "../data/sentiment/sentiment.test.data"
            },

            "BERT_path": {
                "model_type":'Albert',
                "file_path": '../albert_chinese_tiny/',
                "config_path": '../albert_chinese_tiny/',
                "vocab_path": '../albert_chinese_tiny/',
            },

            "training_rule": {
                "continue_train":True, # 是否继续从保存的模型开始训练
                "use_cpu":False,
                "hidden_size":312,
                "max_length": 100, # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.2,
                "num_labels": 2, # 几分类个数
                "show_metric_iter": 3,
                "learning_rate": 1e-4,
                "weight_decay": 1e-2,
                "batch_size": 32
            },

            "result": {
                "model_save_path": '../result/bert_clf_model.bin',
                "config_save_path": '../result/bert_clf_config.json',
                "vocab_save_path": '../result/bert_clf_vocab.txt'
            }
            ,
            "valid_pred_output":{
                "valid_output":'../result/valid.xls',
                "pred_output": '../result/pred.xls'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]