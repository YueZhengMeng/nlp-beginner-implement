import torch
from torch.utils.data import Dataset


class PoetryDataset(Dataset):
    def __init__(self, poetry_list, tokenizer):
        self.poetry_list = poetry_list
        self.tokenizer = tokenizer
        self.input_ids_list = []
        self.output_ids_list = []
        self.labels_list = []

        for poetry in self.poetry_list:
            poetry = poetry.split('，', 1)
            if len(poetry) != 2:
                continue
            self.input_ids_list.append(torch.tensor(self.tokenizer.tokenize(poetry[0])))
            output_ids = self.tokenizer.tokenize(poetry[1])[:-1]
            output_ids.insert(0, self.tokenizer.vocab['<BOS>']["index"])
            self.output_ids_list.append(torch.tensor(output_ids))
            labels = self.tokenizer.tokenize(poetry[1])
            labels[-1] = self.tokenizer.vocab['<EOS>']["index"]
            self.labels_list.append(torch.tensor(labels))

        self.input_length = torch.tensor([len(input_id) for input_id in self.input_ids_list])
        self.output_length = torch.tensor([len(output_id) for output_id in self.output_ids_list])

        pass

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, idx):
        return self.input_ids_list[idx], self.input_length[idx], self.output_ids_list[idx], self.output_length[idx], \
            self.labels_list[idx]


class BatchPadCollator:
    def __call__(self, batch):
        input_ids_list, input_length_list, output_ids_list, output_length_list, label_list = zip(*batch)
        max_input_length = max(input_length_list)
        max_output_length = max(output_length_list)

        input_length = torch.tensor(input_length_list)
        output_length = torch.tensor(output_length_list)

        input_ids = torch.zeros(len(input_ids_list), max_input_length, dtype=torch.int)
        output_ids = torch.zeros(len(output_ids_list), max_output_length, dtype=torch.int)
        # label的数据类型是long，因为CrossEntropyLoss的target需要是long类型
        labels = torch.zeros(len(label_list), max_output_length, dtype=torch.long)
        # input理论上应该是左padding，但是这里是右padding，因为nn.utils.rnn.pack_padded_sequence默认右padding
        # output和label进行右padding
        for i in range(len(input_ids_list)):
            input_ids[i, :len(input_ids_list[i])] = input_ids_list[i]
            output_ids[i, :len(output_ids_list[i])] = output_ids_list[i]
            labels[i, :len(label_list[i])] = label_list[i]

        return input_ids, input_length, output_ids, output_length, labels
