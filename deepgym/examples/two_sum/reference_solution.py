"""Reference solution: hash map approach for O(n) time."""


def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that add up to target.

    Uses a hash map for O(n) time complexity and O(n) space.
    Returns [i, j] with i < j, or [] if no pair exists.
    """
    seen = {}  # value -> index

    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i

    return []
