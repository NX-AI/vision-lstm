class UseModeWrapperException(Exception):
    def __init__(self):
        super().__init__("wrap kappadata.KDDataset into kappadata.ModeWrapper before calling __getitem__")
