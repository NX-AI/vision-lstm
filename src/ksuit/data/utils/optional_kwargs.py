import inspect


def optional_ctx(fn, ctx):
    # returns dict(ctx=ctx) if fn takes a ctx argument
    if ctx is not None and "ctx" in inspect.getfullargspec(fn).args:
        kwargs = dict(ctx=ctx)
    else:
        kwargs = {}
    return kwargs
