import torch

class token_mapping:

    def __init__(self, token_names, inputslices, outputslice):
        self._token_names = token_names
        self.token_inputslices = inputslices
        self.token_outputslice = outputslice

    @property
    def token_names(self):
        return self._token_names

    def input_dim(self, idx):
        input_ids = self.token_inputslices[idx]
        return len(input_ids)

    def output_dim(self, idx):
        oslice = self.token_outputslice[idx]
        return len(oslice)

    def output_dims(self):
        dims = 0
        for oslice in self.token_outputslice:
            dims += len(oslice)
        return dims

    def output_slice(self, idx):
        return self.token_outputslice[idx]

    def create_observation(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = obs["policy"]

        token_lists = []
        for id, slices in enumerate(self.token_inputslices):
            if isinstance(slices, list):
                slices = torch.tensor(slices, dtype=torch.long, device=obs.device)
                self.token_inputslices[id] = slices

            token = obs[:, slices]
            token_lists.append(token)
        return token_lists

