class SentenceDataloader:
    def __init__(self, df, tokenizer, batch_size, data_size=None):
        self.df = df
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        if data_size is not None:
            self.df = self.df[:data_size]
        self.train_step = len(self.df) // batch_size
        self.data_size = len(self.df)
        self.index = 0

    def get_batch(self):
        # 如果index超出范围，重置index
        # 实现多epoch循环取数据
        if self.index + self.batch_size >= len(self.df):
            self.index = 0
        batch_df = self.df.iloc[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        return self.tokenizer.generate_feature(batch_df['Phrase']), batch_df['Sentiment']


if __name__ == '__main__':
    # 测试dataloader
    import pandas as pd
    from tokenizers import BagOfWord

    train_df = pd.read_csv('../Sentiment Analysis on Movie Reviews/train.tsv', sep='\t', header=0)
    tokenizer = BagOfWord()
    dataloader = SentenceDataloader(train_df, tokenizer, 32, 1000)
    x, label = dataloader.get_batch()
    print(x.shape, label.shape)
