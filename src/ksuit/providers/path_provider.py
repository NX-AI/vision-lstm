from pathlib import Path


class PathProvider:
    def __init__(self, output_path: Path, model_path: Path, stage_name: str, stage_id: str):
        self.output_path = output_path
        self.model_path = model_path
        self.stage_name = stage_name
        self.stage_id = stage_id

    @staticmethod
    def _mkdir(path: Path) -> Path:
        path.mkdir(exist_ok=True, parents=True)
        return path

    def get_stage_output_path(self, stage_name: str, stage_id: str, mkdir=True) -> Path:
        stage_output_path = self.output_path / stage_name / stage_id
        return self._mkdir(stage_output_path) if mkdir else stage_output_path

    @property
    def stage_output_path(self) -> Path:
        return self.get_stage_output_path(stage_name=self.stage_name, stage_id=self.stage_id)

    @property
    def stage_output_path_exists(self) -> bool:
        return self.get_stage_output_path(stage_name=self.stage_name, stage_id=self.stage_id, mkdir=False).exists()

    @property
    def logfile_uri(self) -> Path:
        return self.stage_output_path / "log.txt"

    def get_primitive_output_path(self, stage_name: str, stage_id: str) -> Path:
        stage_output_path = self.get_stage_output_path(stage_name=stage_name, stage_id=stage_id)
        return self._mkdir(stage_output_path / "primitive")

    @property
    def primitive_output_path(self) -> Path:
        return self.get_primitive_output_path(stage_name=self.stage_name, stage_id=self.stage_id)

    def get_primitive_config_uri(self, stage_name: str, stage_id: str) -> Path:
        return self.get_primitive_output_path(stage_name=stage_name, stage_id=stage_id) / "config.yaml"

    @property
    def primitive_config_uri(self) -> Path:
        return self.get_primitive_config_uri(stage_name=self.stage_name, stage_id=self.stage_id)

    def get_primitive_entries_uri(self, stage_name: str, stage_id: str) -> Path:
        return self.get_primitive_output_path(stage_name=stage_name, stage_id=stage_id) / "entries.th"

    @property
    def primitive_entries_uri(self) -> Path:
        return self.get_primitive_entries_uri(stage_name=self.stage_name, stage_id=self.stage_id)

    def get_primitive_summary_uri(self, stage_name: str, stage_id: str) -> Path:
        return self.get_primitive_output_path(stage_name=stage_name, stage_id=stage_id) / "summary.yaml"

    @property
    def primitive_summary_uri(self) -> Path:
        return self.get_primitive_summary_uri(stage_name=self.stage_name, stage_id=self.stage_id)

    def get_stage_checkpoint_path(self, stage_name: str, stage_id: str) -> Path:
        return self._mkdir(self.get_stage_output_path(stage_name=stage_name, stage_id=stage_id) / "checkpoints")

    @property
    def checkpoint_path(self) -> Path:
        return self.get_stage_checkpoint_path(stage_name=self.stage_name, stage_id=self.stage_id)
