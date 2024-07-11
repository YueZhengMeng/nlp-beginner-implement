label_set = {'O': 0, 'B-PER': 2, 'I-PER': 4, 'B-ORG': 3, 'I-ORG': 5, 'B-LOC': 1, 'I-LOC': 7, 'B-MISC': 6, 'I-MISC': 8}

def read_conll(data_dir, file_name):
    sentences = []
    labels = []
    with open(data_dir + "/" + file_name, 'r') as f:
        sentence = []
        label = []
        for line in f:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
                continue
            line = line.split()
            sentence.append(line[0])
            label.append(label_set[line[3]])
    # 数据集第一行是列名，需要去掉
    return sentences[1:], labels[1:]


if __name__ == '__main__':
    data_dir = "../conll2003"
    train_file_name = "train.txt"
    val_file_name = "valid.txt"
    test_file_name = "test.txt"

    sentences, labels = read_conll(data_dir, train_file_name)
    print(len(sentences), len(labels))
    print(sentences[:3], labels[:3])
