from math import sqrt

import torch
from torch import nn

CUDA = torch.cuda.is_available()

class Joint(nn.Module):
    def __init__(
            self,
            num_relations,
            num_entities,
            pretrained,
            max_relative_distance=100,
            position_embedding_dim=50,
            conv_window=3,
            hidden_dim=230,
            k=50,
            margin=1.0,
    ):
        super().__init__()
        self.num_relations = num_relations
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained).cuda() if CUDA else torch.FloatTensor(pretrained), padding_idx=0)
        self.pos_dis_embedding = nn.Embedding(max_relative_distance, position_embedding_dim, padding_idx=0)
        self.neg_dis_embedding = nn.Embedding(max_relative_distance, position_embedding_dim, padding_idx=0)
        self.entity_embedding = nn.Embedding(2, position_embedding_dim, padding_idx=0) # 0 for any word that's not an entity

        self.hidden_dim = hidden_dim
        self.conv = nn.Conv1d(300 + position_embedding_dim * 2, hidden_dim, conv_window)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.M = nn.Linear(hidden_dim, num_relations)

        self.W = nn.Linear(hidden_dim, k)

        self.E = nn.Parameter(
            torch.FloatTensor(num_entities, k).uniform_(-6/sqrt(k), 6/sqrt(k))
        )
        self.R = nn.Parameter(
            torch.FloatTensor(num_relations, k).uniform_(-6/sqrt(k), 6/sqrt(k))
        )
        self.margin = margin

    def forward(self, X, positions, h_idx, t_idx, r_idx, h_indices, t_indices, nn=True):
        '''
        X: N * L
        '''
        if nn:
            return self.forward_nn(X, positions, h_idx, t_idx)
        else:
            return self.forward_kg(r_idx, h_idx, t_idx, h_indices, t_indices)

    def forward_nn(self, X, positions, h_idx, t_idx):
        '''
        X: B * N * L
        positions: B * N * L, B * N * L, B * N * L ....
        h_idx, t_idx: B
        '''
        embedding = self.embed(X, *positions) # B * N * L * 400
        B, N, L, E = embedding.shape
        conv_result = self.conv(torch.transpose(embedding.view(-1, L, E), -1, -2)).view(B, N, self.hidden_dim, -1) # B * N * hidden_dim * (L - 2)
        activated_result = self.tanh(conv_result) # B * N * hidden_dim * (L - 2)
        y = torch.max(activated_result, dim=-1)[0] # B * N * hidden_dim
        e = self.tanh(self.W(y)) # B * N * k
        rht = (torch.index_select(self.E, 0, h_idx) - torch.index_select(self.E, 0, t_idx)) # B * k
        e_rht = torch.bmm(e, rht.unsqueeze(2)).squeeze(-1) # B * N
        a = self.softmax(e_rht).unsqueeze(2) # B * N * 1
        rs = torch.bmm(torch.transpose(y, -1, -2), a).squeeze(-1) # B * hidden_dim
        O = self.M(rs) # B * num_relations
        return self.softmax(O)

    
    def forward_kg(self, r_idx, h_idx, t_idx, h_indices, t_indices):
        """
        r_idx: B
        h_idx: B
        t_idx: B
        h_indices: B * N
        t_indices: B * N
        """
        B, N = h_indices.shape
        r = torch.index_select(self.R, 0, r_idx) # B * k
        hs = torch.index_select(self.E, 0, torch.flatten(h_indices)).view(B, N, -1) # B * N * k
        ts = torch.index_select(self.E, 0, torch.flatten(t_indices)).view(B, N, -1) # B * N * k
        Mr = torch.index_select(self.M.weight, 0, r_idx) # B * hidden_dim
        er = self.tanh(self.W(Mr)) # B * k
        rht = hs - ts # B * N * k
        er_rht = torch.bmm(rht, er.unsqueeze(2)).squeeze(-1) # B * N
        b = self.softmax(er_rht) # B * N
        rk = torch.matmul(torch.transpose(rht, -1, -2), b.unsqueeze(2)).squeeze(-1) # B * k
        f = self.margin - torch.norm(
            torch.transpose(rk - self.R.unsqueeze(1).repeat(1, B, 1), -3, -2),
            p=1,
            dim=-1,
        ) # B * num_relations
        pr = self.softmax(f) # B * num_relations

        h = torch.index_select(self.E, 0, h_idx) # B * k
        t = torch.index_select(self.E, 0, t_idx) # B * k
        fh = self.margin - torch.norm(
            torch.transpose(self.E.unsqueeze(1).repeat(1, B, 1) - t - r, -3, -2),
            p=1,
            dim=-1,
        ) # B * num_entities
        ph = self.softmax(fh) # B * num_entities
        ft = self.margin - torch.norm(
            torch.transpose(h - self.E.unsqueeze(1).repeat(1, B, 1) - r, -3, -2),
            p=1,
            dim=-1,
        ) # B * num_entities
        pt = self.softmax(ft) # B * num_entities
        return pr, ph, pt


    def embed(
            self,
            input,
            positive_positions_1,
            positive_positions_2,
            negative_positions_1,
            negative_positions_2,
            entity_positions_1,
            entity_positions_2,
    ):
        """
        input: B * N * L
        positive/negative/entity: B * N * L

        output: B * N * L * 400
        """
        word_embedding = self.embedding(input)
        position_embedding_1 = self.pos_dis_embedding(positive_positions_1) + self.neg_dis_embedding(negative_positions_1) + self.entity_embedding(entity_positions_1)
        position_embedding_2 = self.pos_dis_embedding(positive_positions_2) + self.neg_dis_embedding(negative_positions_2) + self.entity_embedding(entity_positions_2)
        return torch.cat([word_embedding, position_embedding_1, position_embedding_2], dim=-1)
