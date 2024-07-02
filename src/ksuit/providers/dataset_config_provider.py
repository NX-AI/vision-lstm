from ksuit.utils.param_checking import to_path


class DatasetConfigProvider:
    def __init__(
            self,
            global_dataset_paths,
            local_dataset_path=None,
            data_source_modes=None,
    ):
        self._global_dataset_paths = global_dataset_paths
        self._local_dataset_path = local_dataset_path
        self._data_source_modes = data_source_modes

    def get_roots(self, global_root, identifier, local_root=None):
        if global_root is None:
            global_root = self.get_global_dataset_path(identifier)
        else:
            global_root = to_path(global_root)
        if local_root is None:
            source_mode = self.get_data_source_mode(identifier)
            # use local by default
            if source_mode in [None, "local"]:
                local_root = self.local_dataset_path
        else:
            local_root = to_path(local_root)
        return global_root, local_root

    def get_global_dataset_path(self, dataset_identifier):
        assert dataset_identifier in self._global_dataset_paths, \
            f"no path found for identifier {dataset_identifier} -> specify in static_config global_dataset_paths"
        return to_path(self._global_dataset_paths[dataset_identifier], mkdir=False)

    @property
    def local_dataset_path(self):
        return to_path(self._local_dataset_path)

    def get_data_source_mode(self, dataset_identifier):
        if self.local_dataset_path is None:
            return "global"
        if self._data_source_modes is None or dataset_identifier not in self._data_source_modes:
            return None
        data_source_mode = self._data_source_modes[dataset_identifier]
        assert data_source_mode in ["global", "local"], \
            f'data_source_mode {data_source_mode} not in ["global", "local"]'
        return data_source_mode
