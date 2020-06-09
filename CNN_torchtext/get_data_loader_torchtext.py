from torchtext.data import Dataset, BucketIterator, Iterator, Example
from torchtext.vocab import Vectors
import json
import os
from tqdm import trange


class TorchtextSentenceDataset(Dataset):

    def __init__(self, data_path, sentence_field, label_field):
        self.sentence_field = sentence_field
        self.label_field = label_field

        fields = [('sentence', self.sentence_field), ('label', self.label_field)]
        examples = []

        file_read = open(data_path, 'r')
        file_content = json.load(file_read)
        self.sentence_list = []
        self.label_list = []
        for data in file_content:
            self.sentence_list.append(data['sentence'])
            self.label_list.append(int(data['label']))
        file_read.close()

        for index in trange(len(self.sentence_list)):
            examples.append(Example.fromlist([self.sentence_list[index], self.label_list[index]], fields))

        # print(examples[65].sentence)

        for index in range(len(examples)):
            if len(examples[index].sentence) < 4:
                examples[index].sentence.extend('<pad>' for i in range(4-len(examples[index].sentence)))

        # print(examples[65].sentence)

        super().__init__(examples, fields)


def get_iterator(opt, train_data_path, test_data_path, sentence_field, label_field, vectors_path):
    train_dataset = TorchtextSentenceDataset('data/train.json', sentence_field, label_field)
    test_dataset = TorchtextSentenceDataset('data/test.json', sentence_field, label_field)
    cache = 'data/vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name='data/train_word2vec_200.txt', cache=cache)
    sentence_field.build_vocab(train_dataset, min_freq=5, vectors=vectors)

    train_iterator = BucketIterator(train_dataset, batch_size=opt.batch_size,
                                    device='cuda' if opt.cuda else 'cpu', sort_key=lambda x: len(
                                        x.sentence),
                                    sort_within_batch=True, shuffle=True)
    # train_iterator = Iterator(train_dataset, batch_size=opt.batch_size,
    #                                 device='cuda' if opt.cuda else 'cpu', sort_key=lambda x: len(
    #                                     x.sentence),
    #                                 sort_within_batch=True, shuffle=True)
    test_iterator = Iterator(test_dataset, batch_size=opt.batch_size,
                             train=False, sort=False,
                             sort_within_batch=False, shuffle=False,
                             device='cuda' if opt.cuda else 'cpu')

    # for index, data in enumerate(test_iterator):
    #     if index == 65:
    #         a = 0

    return train_iterator, test_iterator, sentence_field.vocab.vectors

# get_iterator('data/train.json', 8, '0', is_test=False)
