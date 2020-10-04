import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self, num_entities, num_relations, knowledge_dim, *args, **kwargs):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.knowledge_dim = knowledge_dim
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, X, heads, tails, *args, **kwargs):
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        assert len(X.shape) == 3
        embedding = self.embed(X, *args, **kwargs)
        encoding = self.encode(embedding)
        assert encoding.shape[-1] == self.num_relations * self.knowledge_dim
        relations_reps = encoding.view(X.shape[0], -1, self.num_relations, self.knowledge_dim)
        scores = self.score_relations(relations_reps, heads, tails)
        assert scores.shape[-1] == self.num_relations
        aggregated = torch.sum(
            scores,
            dim=1,
        )
        return (aggregated.t() / torch.sum(aggregated, dim=-1)).t()

    def predict(self, X, heads, tails, *args, **kwargs):
        return self.forward(
            X,
            heads,
            tails,
            *args,
            **kwargs,
        )

    def embed(self, X, *args, **kwargs):
        raise NotImplementedError

    def encode(self, embedding):
        raise NotImplementedError

    def score_relations(self, relations_reps, heads, tails):
        raise NotImplementedError
