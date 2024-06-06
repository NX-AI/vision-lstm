from .base import SingleModel


class IdentityModel(SingleModel):
    @staticmethod
    def forward(x):
        return x
