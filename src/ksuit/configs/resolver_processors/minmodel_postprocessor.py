import kappaconfig as kc


class MinModelPostProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if isinstance(node, dict):
            if "initializers" in node:
                i = 0
                while i < len(node["initializers"]):
                    if node["initializers"][i]["kind"] == "pretrained_initializer":
                        del node["initializers"][i]
                    else:
                        i += 1
                if len(node["initializers"]) == 0:
                    node.pop("initializers")
