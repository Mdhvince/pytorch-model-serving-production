import re
from typing import Dict, List

import nltk
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class RNNClassifier(nn.Module):
    def __init__(self, config, vocab_size, device, model_path=None, train=False):
        """
        vocab_size = number of words in our vocabulary + 1 (for the 0 padding)
        """
        super(RNNClassifier, self).__init__()

        nltk.download('stopwords')
        output_size = config["MODEL"].getint("output_size")  # number of classes to predict
        embedding_dim = config["MODEL"].getint("embedding_dim")
        self.hidden_dim = config["MODEL"].getint("hidden_dim")
        self.n_layers = config["MODEL"].getint("num_layers")
        self.device = device
        self.config = config

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, self.n_layers, dropout=0.4, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.hidden_dim, output_size)
        self.softmax = nn.Softmax(dim=1)

        if not train:
            _ = self.load(model_path, device=device)
            self.eval()

        self.to(device)

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out[:, -1, :]  # getting the last time step output

        out = self.dropout(lstm_out)
        out = self.fc(out)
        softmax_out = self.softmax(out)
        return softmax_out, hidden

    def init_hidden(self, batch_size):
        """Initializes hidden state"""
        h0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(self.device)
        c0 = torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(self.device)
        hidden = (h0, c0)  # short and long term memory components respectively
        return hidden

    def save(self, model_path: str, epoch: int, loss: float, optimizer: torch.optim.Optimizer, **kwargs):
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": loss,
            "other_params": kwargs
        }, model_path)

    def load(self, model_path: str, optimizer: torch.optim.Optimizer = None, device="cpu"):
        checkpoint: Dict = torch.load(model_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None: optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def instruct(self, model_path, train_loader: DataLoader, valid_loader: DataLoader, optimizer: torch.optim.Optimizer,
                 criterion: nn.CrossEntropyLoss):
        valid_loss_min = np.Inf
        num_epochs = self.config["TRAINING"].getint("num_epochs")
        extra_info = {
            "vocab_size": len(self.embedding.weight),
            "hidden_dim": self.hidden_dim,
            "num_layers": self.n_layers,
            "output_size": self.fc.out_features,
            "embedding_dim": self.embedding.embedding_dim,
            "seq_length": self.config["DEFAULT"].getint("sequence_length")
        }
        for epoch in range(1, num_epochs + 1):
            train_loss = self.learn(train_loader, optimizer, criterion)
            valid_loss = self.validate(valid_loader, criterion)

            train_loss = train_loss / len(train_loader)
            valid_loss = valid_loss / len(valid_loader)
            print(f"Epoch: {epoch} \tTraining Loss: {train_loss} \tValidation Loss: {valid_loss}")

            if valid_loss <= valid_loss_min:
                print(f"Validation loss decreased ({valid_loss_min} --> {valid_loss}).  Saving model ...")
                self.save(model_path, epoch, valid_loss, optimizer, **extra_info)
                valid_loss_min = valid_loss

    def learn(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
              criterion: nn.CrossEntropyLoss) -> float:
        train_loss = 0.0
        clip = self.config["TRAINING"].getfloat("clip")
        self.train()
        h = self.init_hidden(self.config["TRAINING"].getint("batch_size"))
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            h = tuple([each.data for each in h])

            optimizer.zero_grad()
            output, h = self.forward(inputs, h)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), clip)  # to prevent exploding gradient
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        return train_loss

    def validate(self, valid_loader: DataLoader, criterion: nn.CrossEntropyLoss) -> float:
        valid_loss = 0.0
        val_h = self.init_hidden(self.config["TRAINING"].getint("batch_size"))
        self.eval()
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                val_h = tuple([each.data for each in val_h])

                output, val_h = self.forward(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.long())
                valid_loss += val_loss.item() * inputs.size(0)
        return valid_loss

    # Inference functions

    @staticmethod
    def full_predict(model, features, word2idx, seq_length, device):
        features = RNNClassifier.prepare(features, word2idx, seq_length, device)
        response = RNNClassifier.predict(model, features)
        return response

    @staticmethod
    def prepare(text, word2idx, seq_length, device):
        text = RNNClassifier.preprocess_text(text)
        tokenized = RNNClassifier.tokenize_text(text, word2idx)
        features = RNNClassifier.pad_features([tokenized], seq_length)
        return torch.from_numpy(features).to(device)

    @staticmethod
    @torch.no_grad()
    def predict(model, feature_tensor):
        response = {}
        batch_size = feature_tensor.size(0)
        h = model.init_hidden(batch_size)
        output, h = model(feature_tensor, h)
        pred = torch.max(output, 1)

        response["pred_per_class"] = output.squeeze().tolist()
        response["predicted_class"] = pred.indices.item()
        response["predicted_class_proba"] = round(pred.values.item(), 4)
        return response


    @staticmethod
    def preprocess_text(text: str):
        CLEANUP_REGEX = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
        text = CLEANUP_REGEX.sub("", text)  # remove html tags
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
        text = re.sub(r"\s{2,}", " ", text)  # remove extra spaces
        text = text.strip()  # remove leading and trailing spaces
        stop_words = nltk.corpus.stopwords.words("english")
        text = ' '.join([word.strip() for word in text.split() if word.strip() not in stop_words])
        text = ' '.join([nltk.PorterStemmer().stem(word) for word in text.split()])
        text = ' '.join([nltk.WordNetLemmatizer().lemmatize(word) for word in text.split()])
        return text


    @staticmethod
    def tokenize_text(text: str, word2idx: dict):
        return [word2idx.get(word, 0) for word in text.split()]


    @staticmethod
    def pad_features(reviews_ints: List[List[int]], seq_length: int) -> np.ndarray:
        """
        Return features of review_ints, where each review is padded with 0's  or truncated to the input seq_length.
        reviews_ints: list of reviews (i.e [[1, 2, 3], [4, 5, 6, 7], ...])
        seq_length: the length of each sequence (i.e. 200)
        """
        features = np.zeros((len(reviews_ints), seq_length), dtype=int)
        for i, row in enumerate(reviews_ints):
            features[i, -len(row):] = np.array(row)[:seq_length]
        return features
