import kappaconfig as kc


class TestrunPostProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            if parent_accessor == "trainer":
                node.pop("use_torch_compile", None)