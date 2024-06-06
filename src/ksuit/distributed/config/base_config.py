import torch.distributed as dist


class BaseConfig:
    def is_managed(self) -> bool:
        raise NotImplementedError

    def get_local_rank(self) -> int:
        raise NotImplementedError

    def get_num_nodes(self) -> int:
        raise NotImplementedError

    def get_managed_world_size(self) -> int:
        """
        extract world_size from environment variables (e.g. via SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES)
        required for managed runs to derive world_size for initializing process group
        """
        raise NotImplementedError

    def get_managed_rank(self) -> int:
        """
        extract rank from environment variables (e.g. via SLURM_PROCID) required for
        managed runs to derive rank for initializing process group
        """
        raise NotImplementedError

    @staticmethod
    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def get_rank(self) -> int:
        if self.is_distributed():
            return dist.get_rank()
        return 0

    def get_world_size(self) -> int:
        if self.is_distributed():
            return dist.get_world_size()
        return 1

    def is_data_rank0(self) -> bool:
        # data has to be copied in 2 cases
        # - is_local_rank0: single-gpu, multi-gpu, multi-gpu SLURM
        #   - process with is_local_rank0 copies the data
        #   - other processes have to wait for the copying to finish via barrier
        # - get_world_size == 1: SLURM runs that are not using multi-gpu require every process to copy data
        #   - no guarantee that the processes use the same dataset
        #   - avoid race conditions
        return self.is_local_rank0() or self.get_world_size() == 1

    def is_rank0(self) -> bool:
        return self.get_rank() == 0

    def is_local_rank0(self) -> bool:
        return self.get_local_rank() == 0

    def barrier(self) -> None:
        if self.is_distributed():
            dist.barrier()

    def is_own_work(self, idx) -> bool:
        return idx % self.get_world_size() == self.get_rank()

    def get_backend(self) -> str:
        if self.is_distributed():
            return dist.get_backend()
        return None

    def log_distributed_config(self) -> None:
        raise NotImplementedError
