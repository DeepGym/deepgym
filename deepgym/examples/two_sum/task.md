# Two Sum

Write a Python function `two_sum(nums: list[int], target: int) -> list[int]` that returns the indices of two numbers in the list that add up to the target value.

## Requirements

- Each input has exactly one valid solution (when a solution exists).
- You may not use the same element twice.
- Return the indices as a list of two integers `[i, j]` where `i < j`.
- If no valid pair exists, return an empty list `[]`.
- The solution should run in O(n) time complexity.

## Examples

```python
two_sum([2, 7, 11, 15], 9)   # -> [0, 1]
two_sum([3, 2, 4], 6)        # -> [1, 2]
two_sum([3, 3], 6)           # -> [0, 1]
two_sum([1, 2, 3], 10)       # -> []
```

## Signature

```python
def two_sum(nums: list[int], target: int) -> list[int]:
    ...
```
