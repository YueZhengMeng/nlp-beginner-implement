import re


def read_snli(data_dir, file_name):
    """将SNLI数据集解析为前提、假设和标签"""

    def extract_text(s):
        # 删除我们不会使用的信息
        s = re.sub('\\(', '', s)
        s = re.sub('\\)', '', s)
        # 用一个空格替换两个或多个连续的空格
        s = re.sub('\\s{2,}', ' ', s)
        return s.strip()

    print("开始读取数据集： ", file_name)
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    with open(data_dir + "/" + file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    print("数据集 " + file_name + " 读取完毕")
    return premises, hypotheses, labels


if __name__ == '__main__':
    data_dir = "../snli_1.0"
    train_file_name = "snli_1.0_train.txt"
    dev_file_name = "snli_1.0_dev.txt"
    test_file_name = "snli_1.0_test.txt"

    train_data = read_snli(data_dir, train_file_name)
    print(len(train_data[0]), len(train_data[1]), len(train_data[2]))
    print(train_data[0][:3], train_data[1][:3], train_data[2][:3])
