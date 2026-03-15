def top_k_frequent(nums: list, k: int) -> list:
    from collections import Counter

    count = Counter(nums)
    # Sort by frequency descending, then by value ascending for ties
    sorted_items = sorted(count.keys(), key=lambda x: (-count[x], x))
    return sorted_items[:k]
