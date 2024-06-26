def select_with_path(obj, path):
    if path is not None and len(path) > 0:
        for p in path.split("."):
            if isinstance(obj, dict):
                obj = obj[p]
            elif isinstance(obj, list):
                obj = obj[int(p)]
            else:
                obj = getattr(obj, p)
    return obj
