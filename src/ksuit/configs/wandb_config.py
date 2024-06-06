class WandbConfig:
    MODES = ["disabled", "online", "offline"]

    def __init__(self, mode: str, host: str = None, entity: str = None, project: str = None):
        assert mode in self.MODES
        self.mode = mode
        if not self.is_disabled:
            assert host is not None and isinstance(host, str), f"wandb host is required (got '{host}')"
            assert entity is not None and isinstance(entity, str), f"wandb entity is required (got '{project}')"
            assert project is not None and isinstance(project, str), f"wandb project is required (got '{project}')"
        self.host = host
        self.entity = entity
        self.project = project

    @property
    def is_disabled(self) -> bool:
        return self.mode == "disabled"

    @property
    def is_offline(self) -> bool:
        return self.mode == "offline"

    @property
    def is_online(self) -> bool:
        return self.mode == "online"
