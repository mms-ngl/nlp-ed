import os
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

import json
from collections import Counter
from typing import Tuple, List, Any, Dict

from gensim.models import KeyedVectors

import matplotlib.pyplot as plt

torch.manual_seed(1)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print("Using {}.".format(device))


train_file = "../../data/train.jsonl"
dev_file = "../../data/dev.jsonl"

save_path = "../../model/"


pre_embeds_name = "glove-wiki-gigaword-50"

pad_token = "<pad>"
unk_token = "<unk>"

batch_size = 32
training_number = 200


# Data Preparation
class EventDetectionDataset(Dataset):

    def __init__(self,
                 dataset_path:str):
        """
        Args:
            input_file (string): The path to the dataset to be loaded.
        """

        self.dataset_path = dataset_path

        self.data = self.read_dataset()

        # We will initialize it later by calling a function, when we will will need it.
        self.encoded_data = None

    def read_dataset(self):
        tokens, labels = [], []

        max_len = []
        with open(self.dataset_path) as f:
            for line in f:
                data = json.loads(line)

                tokens.append(data["tokens"])
                labels.append(data["labels"])

        assert len(tokens) == len(labels)

        return {"tokens": tokens, "labels": labels}

    def get_encoded_data(self, token_vocab, label_vocab):

        e_tokens, e_labels = [], []

        assert len(self.data["tokens"]) == len(self.data["labels"])

        for idx in range(len(self.data["tokens"])):

            tokens2ids = []
            for token in self.data["tokens"][idx]:

                id_ = token_vocab[token] if token in token_vocab else token_vocab[unk_token]
                tokens2ids.append(id_)

            e_tokens.append(tokens2ids)

            labels2ids = []
            for label in self.data["labels"][idx]:

                id_ = label_vocab[label]
                labels2ids.append(id_)

            e_labels.append(labels2ids)

        self.encoded_data = {
                "e_tokens": e_tokens,
                "e_labels": e_labels
            }

    def __len__(self):
        assert len(self.data["tokens"]) == len(self.data["labels"])
        return len(self.data["tokens"])

    def __getitem__(self, idx):
        return {
                "e_tokens": self.encoded_data["e_tokens"][idx],
                "e_labels": self.encoded_data["e_labels"][idx]
            }

    def get_raw_element(self, idx):
        return {
                "tokens": self.data["tokens"][idx],
                "labels": self.data["labels"][idx]
            }

def build_vocabulary(dataset):
    """Defines the vocabulary to be used. Builds a mapping (word, index) for
    each word in the vocabulary.
    """

    tokens_list = []
    labels_list = []

    for idx in range(len(dataset)):
        tokens_list.extend(dataset.get_raw_element(idx)["tokens"])
        labels_list.extend(dataset.get_raw_element(idx)["labels"])

    tokens_counter = Counter(tokens_list)
    labels_counter = Counter(labels_list)

    token_vocab = {key: index for index, (key, _) in enumerate(tokens_counter.most_common())}
    label_vocab = {key: index for index, (key, _) in enumerate(labels_counter.most_common())}

    token_vocab[unk_token] = len(token_vocab)
    token_vocab[pad_token] = len(token_vocab)

    label_vocab[pad_token] = len(label_vocab)

    return token_vocab, label_vocab


# Getting vectors for tokens in vocabulary from pre-trained Glove
# and storing corresponding vectors for embedding layer.
def embeddings(pre_embeds:KeyedVectors, token_vocab:Dict):

    embeds = torch.randn(len(token_vocab), pre_embeds.vectors.shape[1])

    for i, w in enumerate(token_vocab.keys()):

        if w in pre_embeds:
            vec = pre_embeds[w]
            embeds[i] = torch.tensor(vec)

    embeds[token_vocab[pad_token]] = torch.zeros(pre_embeds.vectors.shape[1])

    return embeds


class ModelParams():
    def __init__(self,
                 token_vocab:Dict,
                 label_vocab:Dict,
                 pre_embeds_name:str=pre_embeds_name):

        pre_embeds_path = os.path.join(os.path.dirname(__file__), save_path + pre_embeds_name)

        pre_embeds = KeyedVectors.load_word2vec_format(pre_embeds_path)

        self.embeds = embeddings(pre_embeds, token_vocab)
        self.padding_idx = token_vocab[pad_token]

        self.embedding_dim = pre_embeds.vectors.shape[1]

        self.hidden_size = 256

        self.num_layers = 1

        self.output_size = len(label_vocab)


# Model Definition
class EventDetectionModel(nn.Module):
    def __init__(self, params):
        super(EventDetectionModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(params.embeds, padding_idx=params.padding_idx)

        self.bilstm = nn.LSTM(params.embedding_dim, params.hidden_size // 2, num_layers=params.num_layers, bidirectional=True)

        self.linear = nn.Linear(params.hidden_size, params.output_size)

    def forward(self, x):
        embeds = self.embedding(x)
        o, (h, c) = self.bilstm(embeds)
        output = self.linear(o)
        return output


# Preparing batching and padding of the train inputs.
def prepare_batch(batch):

    # extract tokens and labels from batch
    tokens = [sample["e_tokens"] for sample in batch]
    labels = [sample["e_labels"] for sample in batch]

    # convert tokens to tensor and pad them
    tokens = pad_sequence(
        [torch.as_tensor(sample) for sample in tokens],
        batch_first=True,
        padding_value=token_vocab[pad_token]
    )

    # convert and pad labels too
    labels = pad_sequence(
        [torch.as_tensor(sample) for sample in labels],
        batch_first=True,
        padding_value=label_vocab[pad_token]
    )

    return {"tokens": tokens, "labels": labels}


# Model Training
class ModelTraining():

    def __init__(self,
                 model:nn.Module,
                 loss_function,
                 optimizer):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_model(self, train_dataset:Dataset, val_dataset:Dataset, epochs:int=1):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            val_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            epochs: the number of times to iterate over train_dataset.
        """

        training_loss, validation_loss = [], []

        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            train_loss = self.train(train_dataset)
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, train_loss))
            training_loss.append(train_loss)

            valid_loss = self.validate(val_dataset)
            print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch+1, valid_loss))
            validation_loss.append(valid_loss)

            # if epoch % 50 == 0 and epoch > 0:
            #     torch.save(self.model.state_dict(), save_path + str(epoch) + '_model_weights.pth')

        return training_loss, validation_loss

    def train(self, train_dataset):
        """
        Args:
            train_dataset: the dataset to use to train the model.

        Returns:
            the average train loss over train_dataset.
        """

        train_loss = 0.0
        self.model.train()

        # for each batch
        for batch in train_dataset:

            tokens = batch['tokens'].to(device)
            labels = batch['labels'].to(device)

            self.optimizer.zero_grad()

            preds = self.model(tokens)
            preds = preds.view(-1, preds.shape[-1])
            labels = labels.view(-1)

            loss = self.loss_function(preds, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.tolist()

        return train_loss / len(train_dataset)

    def validate(self, val_dataset):
        """
        Args:
            val_dataset: the dataset to use to evaluate the model.

        Returns:
            the average validation loss over val_dataset.
        """

        valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():
            for batch in val_dataset:

                tokens = batch['tokens'].to(device)
                labels = batch['labels'].to(device)

                preds = self.model(tokens)
                preds = preds.view(-1, preds.shape[-1])
                labels = labels.view(-1)

                loss = self.loss_function(preds, labels)
                valid_loss += loss.tolist()

        return valid_loss / len(val_dataset)


if __name__ == "__main__":

    # Initializations and functions calls

    trainset = EventDetectionDataset(train_file)
    devset = EventDetectionDataset(dev_file)

    token_vocab, label_vocab = build_vocabulary(trainset)

    params = ModelParams(token_vocab, label_vocab)
    ed_model = EventDetectionModel(params)
    ed_model.to(device)

    trainset.get_encoded_data(token_vocab, label_vocab)
    devset.get_encoded_data(token_vocab, label_vocab)

    train_dataset = DataLoader(
      trainset,
      collate_fn=prepare_batch,
      shuffle=True,
      batch_size=batch_size
    )

    val_dataset = DataLoader(
      devset,
      collate_fn=prepare_batch,
      shuffle=False,
      batch_size=batch_size
    )

    trainer = ModelTraining(
      model = ed_model,
      loss_function = nn.CrossEntropyLoss(ignore_index=label_vocab[pad_token]),
      optimizer = optim.Adam(ed_model.parameters())
    )

    training_loss, validation_loss = trainer.train_model(train_dataset, val_dataset, training_number)

    # Savings
    torch.save(token_vocab, save_path + 'token_vocab.pth')
    torch.save(label_vocab, save_path + 'label_vocab.pth')
    torch.save(ed_model.state_dict(), save_path + 'model_weights.pth')

    plt.plot(range(1, training_number+1), training_loss, label='Train')
    plt.plot(range(1, training_number+1), validation_loss, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()
