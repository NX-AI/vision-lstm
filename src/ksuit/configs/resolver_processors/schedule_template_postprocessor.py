import kappaconfig as kc


# TODO workaround for missing feature of KappaConfig to enable list objects as template
class ScheduleTemplatePostProcessor(kc.Processor):
    """
    resolves nested lists like this:
    schedule:
      schedule:
        - kind: ...
        - kind: ...
    into this:
    schedule:
      - kind: ...
      - kind: ...
    """

    def preorder_process(self, node, trace):
        if isinstance(node, dict):
            for accessor in list(node.keys()):
                subnode = node[accessor]
                if isinstance(subnode, dict) and len(subnode) == 1 and accessor in subnode:
                    node[accessor] = subnode[accessor]
                    return
