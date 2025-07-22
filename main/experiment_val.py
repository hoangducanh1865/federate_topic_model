# %%
from model.ETM import ETM
from evaluation.topic_coherence import TC_on_wikipedia
from trainer.basic_trainer import BasicTrainer
from test_flwr.fed_utils import get_latest_server_model
from test_flwr.fed_utils import split_data
from data.basic_dataset import BasicDataset, RawDataset
DEVICE = "cuda"

# %%
dataset = BasicDataset(dataset_dir= "datasets/20NG", read_labels=True)

# %%
net = ETM(vocab_size = len(dataset.vocab)).to(DEVICE)
trainer = BasicTrainer(net, dataset)

get_latest_server_model(net, max_round = 200)

# %%
import os
import numpy as np
import scipy

def save_top_word(trainer, dir_path, num_top_words = 15):
    top_words = trainer.get_top_words(num_top_words)
    with open(os.path.join(dir_path, f'top_words_{num_top_words}.txt'), 'w') as f:
        for i, words in enumerate(top_words):
            f.write(words + '\n')
    return top_words

def save_theta(trainer, dir_path):
    train_theta, test_theta = trainer.export_theta()
    np.save(os.path.join(dir_path, 'train_theta.npy'), train_theta)
    np.save(os.path.join(dir_path, 'test_theta.npy'), test_theta)
        
    train_argmax_theta = np.argmax(train_theta, axis=1)
    test_argmax_theta = np.argmax(test_theta, axis=1)
    np.save(os.path.join(dir_path, 'train_argmax_theta.npy'), train_argmax_theta)
    np.save(os.path.join(dir_path, 'test_argmax_theta.npy'), test_argmax_theta)
    return train_theta, test_theta

def save_embeddings(trainer, dir_path):
    if hasattr(trainer.model, 'word_embeddings'):
        word_embeddings = trainer.model.word_embeddings.detach().cpu().numpy()
        np.save(os.path.join(dir_path, 'word_embeddings.npy'), word_embeddings)
        print(f'word_embeddings size: {word_embeddings.shape}')

    if hasattr(trainer.model, 'topic_embeddings'):
        topic_embeddings = trainer.model.topic_embeddings.detach().cpu().numpy()
        np.save(os.path.join(dir_path, 'topic_embeddings.npy'),
                topic_embeddings)
        print(f'topic_embeddings size: {topic_embeddings.shape}')

        topic_dist = scipy.spatial.distance.cdist(topic_embeddings, topic_embeddings)
        np.save(os.path.join(dir_path, 'topic_dist.npy'), topic_dist)

    if hasattr(trainer.model, 'group_embeddings'):
        group_embeddings = trainer.model.group_embeddings.detach().cpu().numpy()
        np.save(os.path.join(dir_path, 'group_embeddings.npy'),
                    group_embeddings)
        print(f'group_embeddings size: {group_embeddings.shape}')

        group_dist = scipy.spatial.distance.cdist(group_embeddings, group_embeddings)
        np.save(os.path.join(dir_path, 'group_dist.npy'), group_dist)

    return word_embeddings, topic_embeddings

# %%
save_top_word(trainer, "result")
save_embeddings(trainer, "result")
save_theta(trainer, "result")

# %%
list_score, score = TC_on_wikipedia("result/top_words_15.txt")
print(score)

# %%
from evaluation.topic_diversity import _diversity

print(_diversity(trainer.get_top_words(15)))

# %%
train_theta, test_theta = trainer.export_theta()

from evaluation.clustering import evaluate_clustering
print(evaluate_clustering(test_theta, dataset.test_labels))

# %%
########################### Evaluate ####################################
from topmost import eva

# get theta (doc-topic distributions)
train_theta, test_theta = trainer.export_theta()


# topic diversity
TD = eva._diversity(trainer.get_top_words(15))
print(f"TD: {TD:.5f}")

# clustering
results = eva._clustering(test_theta, dataset.test_labels)
print(results)

# classification
results = eva._cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
print(results)


