from __future__ import annotations

import os

import gensim
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

DEFAULT_BICA_REPO = os.environ.get("GRAPHMED_BICA_MODEL", "bisectgroup/BiCA-base")
batch_size = 1024

# replace with the path to the word2vec file
word2vec_hidden_dim = 300
word2vec_path = "word2vec/GoogleNews-vectors-negative300.bin.gz"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, input_ids=None, attention_mask=None):
        super().__init__()
        self.data = {
            "input_ids": input_ids,
            "att_mask": attention_mask,
        }

    def __len__(self):
        return self.data["input_ids"].size(0)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()
        batch_data = {}
        for key in self.data.keys():
            if self.data[key] is not None:
                batch_data[key] = self.data[key][index]
        return batch_data


class MeanPoolingEncoder(nn.Module):
    def __init__(self, pretrained_repo):
        super().__init__()
        print(f"inherit model weights from {pretrained_repo}")
        self.bert_model = AutoModel.from_pretrained(pretrained_repo)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        data_type = token_embeddings.dtype
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(data_type)
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, att_mask):
        bert_out = self.bert_model(input_ids=input_ids, attention_mask=att_mask)
        sentence_embeddings = self.mean_pooling(bert_out, att_mask)
        return F.normalize(sentence_embeddings, p=2, dim=1)


def _embedding_dim(model, default: int = 768) -> int:
    base_model = getattr(model, "module", model)
    if hasattr(base_model, "get_sentence_embedding_dimension"):
        dim = base_model.get_sentence_embedding_dimension()
        if dim is not None:
            return int(dim)
    config = getattr(getattr(base_model, "bert_model", base_model), "config", None)
    return int(getattr(config, "hidden_size", default))


def load_bica():
    repo = os.environ.get("GRAPHMED_BICA_MODEL", DEFAULT_BICA_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading BiCA encoder from {repo}")
        model = SentenceTransformer(repo, device=str(device))
        model.eval()
        return model, None, device
    except Exception as exc:
        print(f"SentenceTransformer loading failed ({exc}); falling back to AutoModel pooling.")

    model = MeanPoolingEncoder(repo)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    if torch.cuda.device_count() > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    return model, tokenizer, device


def bica_text2embedding(model, tokenizer, device, text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)

    if len(texts) == 0:
        return torch.zeros((0, _embedding_dim(model)))

    if hasattr(model, "encode"):
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        return embeddings.detach().clone().cpu()

    encoding = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    dataset = Dataset(input_ids=encoding.input_ids, attention_mask=encoding.attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_embeddings = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}
            embeddings = model(input_ids=batch["input_ids"], att_mask=batch["att_mask"])
            all_embeddings.append(embeddings)

    return torch.cat(all_embeddings, dim=0).cpu()


def load_word2vec():
    print(f"Loading Google's pre-trained Word2Vec model from {word2vec_path}...")
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model, None, device


def text2embedding_word2vec(model, tokenizer, device, text):
    if type(text) is list:
        text_vector = torch.stack([text2embedding_word2vec(model, tokenizer, device, t) for t in text])
        return text_vector

    words = text.split()
    word_vectors = []

    for word in words:
        try:
            vector = model[word]
            word_vectors.append(vector)
        except KeyError:
            pass

    if word_vectors:
        text_vector = sum(word_vectors) / len(word_vectors)
    else:
        text_vector = np.zeros(word2vec_hidden_dim)

    return torch.Tensor(text_vector)


def load_contriever():
    print("Loading contriever model...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
    model = AutoModel.from_pretrained("facebook/contriever")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def contriever_text2embedding(model, tokenizer, device, text):
    def mean_pooling(token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    try:
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        dataset = Dataset(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        all_embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["att_mask"])
                embeddings = mean_pooling(outputs[0], batch["att_mask"])
                all_embeddings.append(embeddings)
            all_embeddings = torch.cat(all_embeddings, dim=0).cpu()
    except Exception:
        all_embeddings = torch.zeros((0, 1024))

    return all_embeddings


# Backward-compatible aliases for older scripts/checkpoints.
load_sbert = load_bica
sber_text2embedding = bica_text2embedding

load_model = {
    "bica": load_bica,
    "sbert": load_bica,
    "contriever": load_contriever,
    "word2vec": load_word2vec,
}

load_text2embedding = {
    "bica": bica_text2embedding,
    "sbert": bica_text2embedding,
    "contriever": contriever_text2embedding,
    "word2vec": text2embedding_word2vec,
}
