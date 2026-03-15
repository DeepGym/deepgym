# Binary Search

Given a sorted array of integers `nums` and a target value, return the index of `target` if it is present, otherwise return -1.

## Function Signature

```python
def binary_search(nums: list[int], target: int) -> int:
```

## Parameters
- `nums` — a sorted list of distinct integers
- `target` — the value to search for

## Returns
- The index of target in nums, or -1 if not found

## Examples

```
binary_search([-1,0,3,5,9,12], 9) -> 4
binary_search([-1,0,3,5,9,12], 2) -> -1
binary_search([], 5) -> -1
```
