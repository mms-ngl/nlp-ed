### Event Detection
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


device = "cpu"

train_file = "../../data/train.jsonl"
dev_file = "../../data/dev.jsonl"
test_file = "../../data/test.jsonl"


model_folder = "../../model/"

pad_token = "<pad>"
unk_token = "<unk>"


pre_embeddings_name = "glove-wiki-gigaword-50"


### Data Preparation
class EventDetectionDataset(Dataset):

    def __init__(self, 
                 dataset_path:str, 
                 device="cpu"):
        """
        Args:
            input_file (string): The path to the dataset to be loaded.
            device (string): device where to put tensors (cpu or cuda).
        """


        self.dataset_path = dataset_path
        self.device = device

        self.data = self.read_dataset()

        # # we will initialize it later by calling a function, when we will will need it
        # self.encoded_data = None
        self.encoded_data = None

    def read_dataset(self):
        token_data, label_data = [], []

        max_len = []
        with open(self.dataset_path) as f:
            for line in f:
                data = json.loads(line)

                token_data.append(data["tokens"])
                label_data.append(data["labels"])

        assert len(token_data) == len(label_data)

        return {"token_data": token_data, "label_data": label_data}


    def encode_data(self, token_vocab, label_vocab):

        e_token_data, e_label_data = [], []
        
        assert len(self.data["token_data"]) == len(self.data["label_data"])

        for idx in range(len(self.data["token_data"])):

            tokens2ids = []
            for token in self.data["token_data"][idx]:
                id_ = token_vocab[token] if token in token_vocab else token_vocab[unk_token]
                tokens2ids.append(id_)
            e_token_data.append(tokens2ids)


            labels2ids = []
            for label in self.data["label_data"][idx]:
                id_ = label_vocab[label]
                labels2ids.append(id_)
            e_label_data.append(labels2ids)

        self.encoded_data = {"e_token_data": e_token_data, "e_label_data": e_label_data}


    def __len__(self):
        assert len(self.data["token_data"]) == len(self.data["label_data"])
        return len(self.data["token_data"])

    def __getitem__(self, idx):
        return {"e_tokens": self.encoded_data["e_token_data"][idx], "e_labels": self.encoded_data["e_label_data"][idx]}

    def get_raw_element(self, idx):
        return {"tokens": self.data["token_data"][idx], "labels": self.data["label_data"][idx]}


def build_vocabulary(dataset):
    """Defines the vocabulary to be used. Builds a mapping (word, index) for
    each word in the vocabulary.
    """

    token_counter_list = []
    label_counter_list = []

    for idx in range(len(dataset)):
        token_counter_list.extend(dataset.get_raw_element(idx)["tokens"])
        label_counter_list.extend(dataset.get_raw_element(idx)["labels"])

    token_counter = Counter(token_counter_list)
    label_counter = Counter(label_counter_list)

    token_dictionary = {key: index for index, (key, _) in enumerate(token_counter.most_common())}
    label_dictionary = {key: index for index, (key, _) in enumerate(label_counter.most_common())}

    token_dictionary[unk_token] = len(token_dictionary)
    token_dictionary[pad_token] = len(token_dictionary) 

    label_dictionary[pad_token] = len(label_dictionary) 

    return token_dictionary, label_dictionary


# getting vectors for tokens in vocabulary from pre-trained Glove 
# and storing corresponding vectors for embedding layer 
def embeddings(pre_embeddings:KeyedVectors, token_vocab:Dict):

    embeddings = torch.randn(len(token_vocab), pre_embeddings.vectors.shape[1])

    for i, w in enumerate(token_vocab.keys()):

        if w in pre_embeddings:
            vec = pre_embeddings[w]
            embeddings[i] = torch.tensor(vec)
        
    embeddings[token_vocab[pad_token]] = torch.zeros(pre_embeddings.vectors.shape[1])

    return embeddings


class ModelParams():
    def __init__(self, 
                 token_vocab:Dict, 
                 label_vocab:Dict, 
                 pre_embeddings_name:str=pre_embeddings_name):


        pre_embeddings_path = os.path.join(os.path.dirname(__file__), model_folder, pre_embeddings_name)

        pre_embeddings = KeyedVectors.load_word2vec_format(pre_embeddings_path)

        self.embeddings = embeddings(pre_embeddings, token_vocab)
        self.padding_idx = token_vocab[pad_token]

        self.embedding_dim = pre_embeddings.vectors.shape[1]

        self.hidden_size = 256


        self.num_layers = 1

        self.output_size = len(label_vocab)


### Model Definition
class EventDetectionModel(nn.Module):
    def __init__(self, params):
        super(EventDetectionModel, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(params.embeddings, padding_idx=params.padding_idx)

        self.bilstm = nn.LSTM(params.embedding_dim, params.hidden_size // 2, num_layers=params.num_layers, bidirectional=True)

        self.linear = nn.Linear(params.hidden_size, params.output_size)

    
    def forward(self, x):
        embeddings = self.embedding(x)
        o, (h, c) = self.bilstm(embeddings)
        output = self.linear(o)
        return output

# preparing batching and padding of the train inputs
def prepare_batch(batch):

    # extract tokens and labels from batch
    x = [sample["e_tokens"] for sample in batch]
    y = [sample["e_labels"] for sample in batch]

    # convert tokens to tensor and pad them
    x = pad_sequence(
        [torch.as_tensor(sample) for sample in x],
        batch_first=True,
        padding_value=token_vocab[pad_token]
    )

    # convert and pad labels too
    y = pad_sequence(
        [torch.as_tensor(sample) for sample in y],
        batch_first=True,
        padding_value=label_vocab[pad_token]
    )

    return {"x": x, "y": y}


### Model Training 
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
        # train_loss = 0.0
        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))


            train_loss = self.train(train_dataset)
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, train_loss))
            training_loss.append(train_loss)


            valid_loss = self.validate(val_dataset)
            print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch+1, valid_loss))
            validation_loss.append(valid_loss)


            with open('train_losses.txt', 'w') as f:
                f.write(str(training_loss))

            with open('valid_losses.txt', 'w') as f:
                f.write(str(validation_loss))

            if epoch % 10 == 0 and epoch > 0:
                torch.save(self.model.state_dict(), model_folder + str(epoch) + '_model_weights.pth')

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
            x = batch['x']
            y = batch['y']

            self.optimizer.zero_grad()

            preds = self.model(x)
            preds = preds.view(-1, preds.shape[-1])
            y = y.view(-1)
            
            loss = self.loss_function(preds, y)
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
                x = batch['x']
                y = batch['y']

                preds = self.model(x)
                preds = preds.view(-1, preds.shape[-1])
                y = y.view(-1)

                loss = self.loss_function(preds, y)
                valid_loss += loss.tolist()
        
        return valid_loss / len(val_dataset)

if __name__ == "__main__":

    # Initializations and functions calls

    trainset = EventDetectionDataset(train_file, device=device)
    devset = EventDetectionDataset(dev_file, device=device)


    token_vocab, label_vocab = build_vocabulary(trainset)
    torch.save(token_vocab, model_folder + 'token_vocab.pth')
    torch.save(label_vocab, model_folder + 'label_vocab.pth')


    params = ModelParams(token_vocab, label_vocab)

    ed_model = EventDetectionModel(params).to(device)


    trainset.encode_data(token_vocab, label_vocab)
    devset.encode_data(token_vocab, label_vocab)


    # preparing inputs for a model
    batch_sizes = 32
    training_number = 200

    train_dataset = DataLoader(
      trainset, 
      collate_fn=prepare_batch,
      shuffle=True,
      batch_size=batch_sizes
    )

    val_dataset = DataLoader(
      devset, 
      collate_fn=prepare_batch,
      shuffle=False,
      batch_size=batch_sizes
    )


    trainer = ModelTraining(
      model = ed_model,
      loss_function = nn.CrossEntropyLoss(ignore_index=label_vocab[pad_token]),
      optimizer = optim.Adam(ed_model.parameters())
    )

    training_loss, validation_loss = trainer.train_model(train_dataset, val_dataset, training_number)

    plt.plot(range(1, training_number+1), training_loss, label='Train')
    plt.plot(range(1, training_number+1), validation_loss, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()

    torch.save(ed_model.state_dict(), model_folder + 'model_weights.pth')
