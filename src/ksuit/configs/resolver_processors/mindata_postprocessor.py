import kappaconfig as kc


class MinDataPostProcessor(kc.Processor):
    """
    hyperparams for specific properties in the dictionary and replace it such that the training duration is
    limited to a minimal configuration
    """

    def __init__(self, effective_batch_size, updates_per_epoch):
        self.effective_batch_size = effective_batch_size
        self.updates_per_epoch = updates_per_epoch

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            # datasets
            if parent_accessor == "datasets":
                for key in node.keys():
                    wrappers = [
                        # shuffle dataset
                        dict(
                            kind="shuffle_wrapper",
                            seed=0,
                        ),
                        # create subset
                        dict(
                            kind="subset_wrapper",
                            end_index=self.effective_batch_size * self.updates_per_epoch + 1,
                        ),
                    ]
                    # append mindata wrappers to (possible existing) dataset_wrappers
                    if "dataset_wrappers" in node[key]:
                        node[key]["dataset_wrappers"] += wrappers
                    else:
                        assert isinstance(node[key], dict), (
                            "found non-dict value inside 'datasets' node -> probably wrong template "
                            "parameter (e.g. template.version instead of template.vars.version)"
                        )
                        node[key]["dataset_wrappers"] = wrappers
            elif parent_accessor == "effective_batch_size":
                parent[parent_accessor] = min(parent[parent_accessor], self.effective_batch_size)
            elif parent_accessor == "optim":
                # decrease lr scaling (e.g. to avoid errors when max_lr < min_lr when using a min_lr with cosine decay)
                parent[parent_accessor]["lr_scaler"] = dict(
                    kind="linear_lr_scaler",
                    divisor=self.effective_batch_size,
                )
