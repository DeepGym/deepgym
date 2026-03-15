# Longest Consecutive Sequence

Given an unsorted array of integers `nums`, return the length of the longest consecutive elements sequence.

## Function Signature

```python
def longest_consecutive(nums: list[int]) -> int:
```

## Parameters
- `nums` — an unsorted list of integers

## Returns
- The length of the longest consecutive sequence

## Examples

```
longest_consecutive([100, 4, 200, 1, 3, 2]) -> 4  # [1,2,3,4]
longest_consecutive([0, 3, 7, 2, 5, 8, 4, 6, 0, 1]) -> 9  # [0..8]
longest_consecutive([]) -> 0
```
