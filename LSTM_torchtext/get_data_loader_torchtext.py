from torchtext.data import Dataset, BucketIterator, Iterator, Example
from torchtext.vocab import Vectors
import json
import os
from tqdm import trange


class SentenceDataset(Dataset):

    def __init__(self, data_path, sentence_field, label_field):

        fields = [('sentence', sentence_field), ('label', label_field)]
        examples = []

        with open(data_path, 'r') as f_json:
            file_content = json.load(f_json)
            self.sentence_list = []
            self.label_list = []
            for data in file_content:
                self.sentence_list.append(data['sentence'])
                self.label_list.append(data['label'])
        
        for index in trange(len(self.sentence_list)):
            examples.append(Example.fromlist([self.sentence_list[index], self.label_list[index]], fields))

        super().__init__(examples, fields)

    @staticmethod
    def sort_key(input):
        return len(input.sentence)


def get_iterator(opt, train_data_path, test_data_path, sentence_field, label_field, vector_path):
    train_dataset = SentenceDataset(train_data_path, sentence_field, label_field)
    test_dataset = SentenceDataset(test_data_path, sentence_field, label_field)

    cache = 'data/vector_cache'
    if not os.path.exists(cache):
        os.mkdir(cache)
    vectors = Vectors(name=vector_path, cache=cache)
    sentence_field.build_vocab(train_dataset, min_freq=opt.min_freq, vectors=vectors)

    train_iterator = None
    test_iterator = None
    if opt.iterator_type == 'bucket':
        train_iterator = BucketIterator(train_dataset, batch_size=opt.batch_size,
                                        device='cuda' if opt.cuda else 'cpu',
                                        sort_within_batch=True, shuffle=True)
    elif opt.iterator_type == 'iterator':
        train_iterator = Iterator(train_dataset, batch_size=opt.batch_size,
                                  device='cuda' if opt.cuda else 'cpu',
                                  sort_within_batch=True, shuffle=True)

    test_iterator = Iterator(test_dataset, batch_size=opt.batch_size,
                             device='cuda' if opt.cuda else 'cpu', train=False,
                             shuffle=False, sort=False, sort_within_batch=True)
    
    # for data in test_iterator:
    #     a = 1
    
    return train_iterator, test_iterator, sentence_field.vocab.vectors

