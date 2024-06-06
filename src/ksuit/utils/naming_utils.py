def join_names(name1, name2):
    if name1 is None:
        return name2
    assert name2 is not None
    return f"{name1}.{name2}"


def pascal_to_snake(pascal_case: str) -> str:
    """
    convert pascal/camel to snake case https://learn.microsoft.com/en-us/visualstudio/code-quality/ca1709?view=vs-2022
    "By convention, two-letter acronyms use all uppercase letters,
    and acronyms of three or more characters use Pascal casing."
    """
    if len(pascal_case) == 0:
        return ""
    snake_case = [pascal_case[0].lower()]
    upper_counter = 0
    for i in range(1, len(pascal_case)):
        if pascal_case[i].islower() or pascal_case[i].isdigit():
            snake_case += [pascal_case[i]]
            upper_counter = 0
        else:
            if upper_counter == 2:
                upper_counter = 0
            if upper_counter == 0:
                snake_case += ["_"]
            snake_case += [pascal_case[i].lower()]
            upper_counter += 1
    return "".join(snake_case)


def lower_type_name(obj):
    return type(obj).__name__.lower()


def snake_type_name(obj):
    # convert a type name to snake case
    # preferably use module name as class names are often custom names (e.g. KoLeoLoss)
    # e.g. KoLeoLoss would be converted to ko_leo_loss but if the module is called koleo_loss it will be preferred
    snake = pascal_to_snake(type(obj).__name__)
    module = type(obj).__module__.split(".")[-1]
    if snake.replace("_", "") == module.replace("_", ""):
        return module
    return snake
