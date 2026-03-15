def level_order(tree: list) -> list:
    if not tree or tree[0] is None:
        return []
    result = []
    level_start = 0
    level_size = 1
    while level_start < len(tree):
        level = []
        for i in range(level_start, min(level_start + level_size, len(tree))):
            if tree[i] is not None:
                level.append(tree[i])
        if not level:
            break
        result.append(level)
        level_start += level_size
        level_size *= 2
    return result
