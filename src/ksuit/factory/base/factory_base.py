import logging
from functools import partial


class FactoryBase:
    def __init__(self):
        self.logger = logging.getLogger(type(self).__name__)

    def create(self, obj_or_kwargs, collate_fn=None, **kwargs):
        # create a single object
        if obj_or_kwargs is None:
            return None

        # create an object from a collection of objects
        # e.g. create a transform object that consists of multiple objects packed into a compose transform
        if collate_fn is not None:
            assert isinstance(obj_or_kwargs, list)
            objs = [self.create(obj_or_kwargs[i]) for i in range(len(obj_or_kwargs))]
            if len(objs) == 1:
                obj = objs[0]
            else:
                obj = collate_fn(objs)
            return obj

        #
        if isinstance(obj_or_kwargs, dict):
            if len(obj_or_kwargs) == 0:
                return None
            return self.instantiate(**obj_or_kwargs, **kwargs)

        # check obj_or_kwargs was already instantiated
        if not isinstance(obj_or_kwargs, (partial, type)):
            return obj_or_kwargs

        # check kwargs overlap
        if isinstance(obj_or_kwargs, partial):
            # check for duplicate kwargs (e.g partial(Obj, key=5)(key=3) -> key is passed twice)
            for key in kwargs.keys():
                assert key not in obj_or_kwargs.keywords, f"got multiple values for keyword argument {key}"
        # instantiate
        return obj_or_kwargs(**kwargs)

    def create_list(self, collection, collate_fn=None, **kwargs):
        if collection is None:
            return []
        if isinstance(collection, list):
            objs = []
            for ckwargs in collection:
                if isinstance(ckwargs, dict):
                    objs.append(self.create(ckwargs, **kwargs))
                elif isinstance(ckwargs, (partial, type)):
                    objs.append(self.create(ckwargs, **kwargs))
                else:
                    raise NotImplementedError
        else:
            raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected list)")
        if collate_fn is not None:
            return collate_fn(objs)
        return objs

    def create_dict(self, collection, collate_fn=None, **kwargs):
        if collection is None:
            return {}
        if isinstance(collection, dict):
            objs = {
                key: self.create(ckwargs, **kwargs)
                for key, ckwargs in collection.items()
            }
        else:
            raise NotImplementedError(f"invalid collection type {type(collection).__name__} (expected dict)")
        if collate_fn is not None:
            return collate_fn(objs)
        return objs

    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        raise NotImplementedError
