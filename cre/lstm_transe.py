from math import sqrt
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

import torch
from torch import nn

from cre.base_model import BaseModel

CUDA = torch.cuda.is_available()

class LstmTranse(BaseModel):
    def __init__(
            self,
            num_entities,
            num_relations,
            knowledge_dim,
            pretrained,
            max_relative_distance=100,
            position_embedding_dim=50,
            hidden_dim=230,
            conv_window=3,
    ):
        super().__init__(num_entities, num_relations, knowledge_dim)
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained).cuda() if CUDA else torch.FloatTensor(pretrained), padding_idx=0)
        self.head_position_embedding = nn.Embedding(max_relative_distance * 2 + 1, position_embedding_dim)
        self.tail_position_embedding = nn.Embedding(max_relative_distance * 2 + 1, position_embedding_dim)
        self.max_relative_distance = max_relative_distance

        self.lstm = nn.LSTM(
            input_size=self.embedding.weight.shape[-1] + position_embedding_dim * 2,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.W = nn.Linear(hidden_dim, self.num_relations * self.knowledge_dim)
        self.E = nn.Parameter(
            torch.FloatTensor(num_entities, knowledge_dim).uniform_(-6/sqrt(knowledge_dim), 6/sqrt(knowledge_dim))
        )
        self.hidden_dim = hidden_dim

    def embed(self, X, *args, **kwargs):
        heads = kwargs['h_idx']
        tails = kwargs['t_idx']
        word_embedding = self.embedding(X)
        sent_len = X.shape[-1]
        pos_wrt_heads = [
            [
                [self.get_pos_idx(i, head) for i in range(sent_len)]
                for head in batch
            ]
            for batch in heads
        ]
        pos_wrt_tails = [
            [
                [self.get_pos_idx(i, tail) for i in range(sent_len)]
                for tail in batch
            ]
            for batch in tails
        ]
        head_pos_embeddings = self.head_position_embedding(
            torch.LongTensor(pos_wrt_heads).cuda() if CUDA else torch.LongTensor(pos_wrt_heads),
        )
        tail_pos_embeddings = self.tail_position_embedding(
            torch.LongTensor(pos_wrt_tails).cuda() if CUDA else torch.LongTensor(pos_wrt_tails),
        )
        return torch.cat([word_embedding, head_pos_embeddings, tail_pos_embeddings], dim=-1)

    def encode(self, embedding):
        B, N, L, I = embedding.shape
        reshaped_embedding = embedding.view(-1, L, I) # BN * L * I
        lstm_result = self.lstm(reshaped_embedding)[0].view(B, N, -1, self.hidden_dim) # B * N * L * E
        pool_result = torch.max(self.tanh(lstm_result), dim=-2)[0] # B * N * E
        return self.W(pool_result)

    def score_relations(self, relations_reps, heads, tails):
        hs = torch.index_select(self.E, 0, torch.LongTensor(heads).cuda() if CUDA else torch.LongTensor(heads)) # B * E
        ts = torch.index_select(self.E, 0, torch.LongTensor(tails).cuda() if CUDA else torch.LongTensor(tails)) # B * E
        deltas = hs + torch.transpose(relations_reps, 0, -2) - ts # R * N * B * E
        distances = torch.norm(deltas, p=2, dim=-1) # R * N * B
        distances = torch.transpose(distances, 0, 1) # N * R * B
        probs = self.tanh(distances)
        return torch.transpose(torch.transpose(1 - probs, 0, 1), 0, 2)
    
    def get_pos_idx(self, i, entity_idx):
        if i == entity_idx:
            return 2 * self.max_relative_distance
        elif i > entity_idx:
            return min(self.max_relative_distance - 1, i - entity_idx)
        else:
            return min(self.max_relative_distance - 1, entity_idx - i) + self.max_relative_distance
