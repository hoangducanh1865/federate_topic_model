from data.preprocess import Preprocess
from data.basic_dataset import RawDataset, BasicDataset
from typing import List
from utils._utils import file_utils

# get vocab from multiple datasets
def get_all_vocab(dirs: List[str]) -> List[str]:
    '''
        x = get_all_vocab(["../../data/20NG", "../../data/IMDB"]) -> List words as global vocab
    '''
    all_vocab_set = set()
    for dir in dirs:
        vocab = file_utils.read_text(f'{dir}/vocab.txt')
        all_vocab_set.update(word for word in vocab)  # Sử dụng update thay vì add
    result = list(all_vocab_set)
    result.sort()
    return result

# split data from 1 datasets
def split_data(dir:str, num_split:int, vocab = None, batch_size = 200, device = "cuda") -> List[RawDataset]:
    '''
        datasets = split_data("../../data/20NG", 2, vocab = x)
        -> list RawDataset from a BasicDataset
    '''
    dataset = BasicDataset(dir, batch_size=batch_size, device = device)

    train_texts = dataset.train_texts
    if vocab is None:
        vocab = dataset.vocab
        
    num_sample = int(len(train_texts) / num_split)

    datasets = []
    for i in range(num_split):
        dataset = RawDataset(train_texts[(i * num_sample) : ((i + 1) * num_sample)], vocab = vocab, device = device)
        datasets.append(dataset)
    
    return datasets

    