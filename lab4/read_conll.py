label_set = {'O': 0, 'B-LOC': 1, 'B-PER': 2, 'B-ORG': 3, 'I-PER': 4, 'I-ORG': 5, 'I-LOC': 7, 'B-MISC': 6, 'I-MISC': 8}
label_count = [170523, 7140, 6600, 6321, 4528, 3704, 3438, 1157, 1155]
label_weight = [0.001976221319389293, 0.047197645384624703, 0.051059270916094, 0.05331295491950964, 0.07442384895013703, 0.09098034234509192, 0.09801954277086108, 0.2912629110166123, 0.29176726237768]

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

    label_count = [0] * 9
    for label in labels:
        for l in label:
            label_count[l] += 1

    label_weight = [1 / c for c in label_count]

    label_weight_sum = sum(label_weight)
    label_weight = [w / label_weight_sum for w in label_weight]

    print(label_count)
    print(label_weight)
