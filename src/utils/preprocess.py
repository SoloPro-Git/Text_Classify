import pandas as pd
from sklearn.model_selection import train_test_split
from config import Config


class Preprocess():

    def __init__(self, config):
        self.config = config
        origin_data_path = self.config.get("data_preprocess_path", "data_path")

        self.data = pd.read_excel(origin_data_path, names=["text", "label"])

    def process(self):
        # train_data, test_data = train_test_split(self.data, stratify=self.data["label"])
        train_data, test_data = train_test_split(self.data)

        train_path = self.config.get("data_preprocess_path", "train_path")
        test_path = self.config.get("data_preprocess_path", "test_path")
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)

    def label_num(self):
        return len(self.data['label'].unique())


if __name__ == '__main__':
    preprocess = Preprocess(config=Config())
    preprocess.process()
