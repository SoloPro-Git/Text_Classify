from predict import transformers_bert_binary_classification as Model
import os

os.chdir(r'src/')
model = Model()
print(model.predict("酒店的服务人员服务周到，细致，给人一种宾至如归的感觉"))  # 1
print(1)