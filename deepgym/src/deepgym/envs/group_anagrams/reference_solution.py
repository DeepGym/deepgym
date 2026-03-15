def group_anagrams(strs: list) -> list:
    from collections import defaultdict

    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    result = [sorted(g) for g in groups.values()]
    result.sort(key=lambda g: g[0])
    return result
