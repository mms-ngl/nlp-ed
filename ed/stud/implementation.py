import os
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from typing import List

from model import Model
from stud.ed import ModelParams
from stud.ed import EventDetectionModel


pad_token = "<pad>"
unk_token = "<unk>"

def build_model(device: str) -> Model:

    model_folder = "../../model/"
    model_name = "model_weights.pth"
    token_vocab_name = "token_vocab.pth"
    label_vocab_name = "label_vocab.pth"

    model_path = os.path.join(os.path.dirname(__file__), model_folder + model_name)
    token_vocab_path = os.path.join(os.path.dirname(__file__), model_folder + token_vocab_name)
    label_vocab_path = os.path.join(os.path.dirname(__file__), model_folder + label_vocab_name)

    return StudentModel(model_path, token_vocab_path, label_vocab_path, device)


class StudentModel(Model):

    def __init__(self, 
                 model_path: str, 
                 token_vocab_path: str,
                 label_vocab_path: str,
                 device: str="cpu"):

        # Specifying the device type responsible to load a tensor into memory.
        self.device = torch.device(device)

        self.token_vocab = torch.load(token_vocab_path, map_location=self.device)
        self.label_vocab = torch.load(label_vocab_path, map_location=self.device)

        params = ModelParams(self.token_vocab, self.label_vocab)

        self.ed_model = EventDetectionModel(params).to(self.device)

        self.ed_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Setting the model in evaluation mode.
        self.ed_model.eval()


    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        # Encode tokens to equivalent numercal values, since model understand only numerical values.
        batch = self.encode_inputs(tokens)

        """
        Inputs need to be padded and be in tensor, since model requires same length batch samples
        and tensor values, samples will be filled out with padding_value.
        """
        inputs = pad_sequence(
            [torch.as_tensor(sample) for sample in batch],
            batch_first=True,
            padding_value=self.token_vocab[pad_token]
        )

        # Disabling computing gradients
        with torch.no_grad():
            predictions = self.ed_model(inputs)
        
        predictions = torch.argmax(predictions, -1)

        # Determining padding values and setting their indexes to False.
        valid_indices = inputs != self.token_vocab[pad_token]

        # Getting rid of predictions for padding inputs w.r. to valid_indices.
        valid_predictions = [pred[valid_indices[idx]] for idx, pred in enumerate(predictions)]

        outputs = self.decode_outputs(valid_predictions)

        return outputs

    """
    Converting input token values to model understandible numberical values.  
    """
    def encode_inputs(self, tokens: List[List[str]]) -> List[List[int]]:
        inputs = []

        for sample in tokens:
            e_tokens = []
            for token in sample:
                idx = self.token_vocab[token] if token in self.token_vocab else self.token_vocab[unk_token]
                e_tokens.append(idx)
                
            inputs.append(e_tokens)

        return inputs

    """
    Converting numberical prediction values to human understandible label values. 
    """
    def decode_outputs(self, predictions: List[List[int]]) -> List[List[str]]:
        outputs = []

        for sample in predictions:
            labels = []
            for idx in sample:
                label = list(self.label_vocab.keys())[list(self.label_vocab.values()).index(idx)]
                labels.append(label)
            outputs.append(labels)

        return outputs