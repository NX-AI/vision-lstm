import torch

from .base import SingleModel


class TorchhubModel(SingleModel):
    def __init__(self, repo, model, load_kwargs=None, source="github", **kwargs):
        super().__init__(**kwargs)
        self.repo = repo
        self.model = model
        self.source = source
        self.load_kwargs = load_kwargs or {}
        self.model = torch.hub.load(repo_or_dir=repo, model=model, source=source, **self.load_kwargs)

    def forward(self, x):
        return self.model(x)

    def classify(self, x):
        return dict(main=self.model(x))
