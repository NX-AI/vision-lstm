import kappaconfig as kc

from ksuit.utils.amp_utils import FLOAT32_ALIASES, FLOAT16_ALIASES, BFLOAT16_ALIASES


class PrecisionPreProcessor(kc.Processor):
    def preorder_process(self, node, trace):
        if len(trace) == 0:
            return
        parent, parent_accessor = trace[-1]
        if isinstance(parent_accessor, str):
            if parent_accessor == "precision":
                # replace precision
                actual = parent[parent_accessor].value
                if actual in FLOAT32_ALIASES:
                    precision = "float32"
                elif actual in FLOAT16_ALIASES:
                    precision = "float16"
                elif actual in BFLOAT16_ALIASES:
                    precision = "bfloat16"
                else:
                    raise NotImplementedError

                parent[parent_accessor] = kc.from_primitive(precision)
