"""Reference solution: merge sort implementation."""


def sort_list(lst: list[int]) -> list[int]:
    """Sort a list of integers in ascending order using merge sort.

    Time complexity: O(n log n)
    Space complexity: O(n)
    """
    if len(lst) <= 1:
        return lst[:]

    mid = len(lst) // 2
    left = sort_list(lst[:mid])
    right = sort_list(lst[mid:])

    return _merge(left, right)


def _merge(left: list[int], right: list[int]) -> list[int]:
    """Merge two sorted lists into one sorted list."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
