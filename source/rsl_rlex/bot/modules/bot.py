import torch
import torch.nn as nn
from .bot_utils import MixedTransformerEncoder
from .mappings import token_mapping

class ObsTokenizer(torch.nn.Module):
    def __init__(self, mapping: token_mapping, output_dim):
        super(ObsTokenizer, self).__init__()

        self.mapping = mapping
        self.output_dim = output_dim

        self.zero_token = torch.nn.Parameter(torch.zeros(1, output_dim), requires_grad=False)

        base = lambda input_dim : torch.nn.Sequential(torch.nn.Linear(input_dim, output_dim))
        self.tokenizers = torch.nn.ModuleDict()
        for id, name in enumerate(mapping.token_names):
            self.tokenizers[name] = base(mapping.input_dim(id))

    def forward(self, x):
        x = self.mapping.create_observation(x)
        outputs = []
        for id, name in enumerate(self.mapping.token_names):
            inputs = x[id]
            if inputs.shape[-1] == 0:
                outputs.append(self.zero_token.expand(*inputs.shape[:-1], -1).unsqueeze(1))
            else:
                outputs.append(self.tokenizers[name](inputs).unsqueeze(1))
        return torch.cat(outputs, dim=1)

class ActionDetokenizer(torch.nn.Module):
    def __init__(self, mapping: token_mapping, embedding_dim, global_input=False):
        super(ActionDetokenizer, self).__init__()

        self.mapping = mapping
        self.nbodies = len(mapping.token_names)
        self.embedding_dim = embedding_dim
        self.action_dim = mapping.output_dims()

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(self.action_dim)
        else:
            for id, name in enumerate(mapping.token_names):
                self.detokenizers[name] = base(mapping.output_dim(id))

    def forward(self, x):
        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x)

        action = torch.zeros(x.shape[0], self.action_dim).to(x.device)
        for id, name in enumerate(self.mapping.token_names):
            curr_action = self.detokenizers[name](x[:, id, :])
            oslice = self.mapping.output_slice(id)
            if 1 == len(oslice):
                action[:, oslice[0]: oslice[0] + 1] = curr_action
            elif 2 == len(oslice):
                action[:, oslice[1]: oslice[0]] = curr_action
            else:
                continue

        return action

class ValueDetokenizer(torch.nn.Module):
    def __init__(self, mapping: token_mapping, embedding_dim, global_input=False):
        super(ValueDetokenizer, self).__init__()

        self.mapping = mapping
        self.nbodies = len(mapping.token_names)

        self.embedding_dim = embedding_dim

        base = lambda output_dim : torch.nn.Linear(embedding_dim, output_dim)
        self.detokenizers = torch.nn.ModuleDict()
        if global_input:
            self.detokenizers['global'] = base(1)
        else:
            for id, name in enumerate(mapping.token_names):
                self.detokenizers[name] = base(1)

    def forward(self, x):
        if 'global' in self.detokenizers:
            return self.detokenizers['global'](x)

        values = torch.zeros((x.shape[0], x.shape[1]), device=x.device)
        for id, name in enumerate(self.mapping.token_names):
            values[:, id] = self.detokenizers[name](x[:,id,:]).squeeze(-1)
        return torch.mean(values, dim=1, keepdim=True)

class BodyActor(nn.Module):
    def __init__(self, mapping: token_mapping, net, embedding_dim, global_input=False, activation=None):
        super(BodyActor, self).__init__()

        self.tokenizer = ObsTokenizer(mapping, embedding_dim)
        self.net = net
        self.detokenizer = ActionDetokenizer(mapping, embedding_dim, global_input)
        self.activation = activation

    def forward(self, x):
        x = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)

        x = self.net(x)

        x = self.detokenizer(x)

        if self.activation is not None:
            x = self.activation(x)
        return x

class BodyCritic(nn.Module):
    def __init__(self, mapping: token_mapping, net, embedding_dim, global_input=False):
        super(BodyCritic, self).__init__()

        self.tokenizer = ObsTokenizer(mapping, embedding_dim)
        self.net = net
        self.detokenizer = ValueDetokenizer(mapping, embedding_dim, global_input)

    def forward(self, x):

        x = self.tokenizer(x) # (B, nbodies, lookback_steps, embedding_dim)
        x = self.net(x)
        x = self.detokenizer(x)
        return x

class BodyTransformer(nn.Module):
    def __init__(self, shortest_path_matrix, input_dim, dim_feedforward=256, nhead=6, num_layers=3, is_mixed=True, first_hard_layer=0):
        super(BodyTransformer, self).__init__()

        adjacency_matrix = shortest_path_matrix < 2

        self.nbodies = adjacency_matrix.shape[0]

        # We assume (B x nbodies x input_dim) batches
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True, dropout=0.)
        self.encoder = MixedTransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embed_absolute_position = nn.Embedding(self.nbodies, embedding_dim=input_dim)

        self.is_mixed = is_mixed

        self.first_hard_layer = first_hard_layer

        self.register_buffer('adjacency_matrix', adjacency_matrix)

        self.init_weights()

    def forward(self, x):
        limb_indices = torch.arange(0, self.nbodies, device=x.device)
        limb_idx_embedding = self.embed_absolute_position(limb_indices)
        x = x + limb_idx_embedding
        x = self.encoder(x, mask=~self.adjacency_matrix, is_mixed=self.is_mixed, return_intermediate=False, first_hard_layer=self.first_hard_layer)
        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_weights(self):
        self.apply(self._init_weights)

