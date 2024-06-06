import kappaconfig as kc


class RemoveLargeCollectionsProcessor(kc.Processor):
    """
    remove large list/dicts for prettier storing of the resolved yaml
    """

    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(node, list) and len(node) > 100:
            parent[parent_accessor] = f"list with length {len(node)}"
        if isinstance(node, list) and len(node) > 100:
            parent[parent_accessor] = f"dict with length {len(node)}"
