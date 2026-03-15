# Rotate Array

Given an integer array `nums`, rotate the array to the right by `k` steps.

## Function Signature

```python
def rotate_array(nums: list[int], k: int) -> list[int]:
```

## Parameters
- `nums` — a list of integers
- `k` — number of positions to rotate right

## Returns
- The rotated array

## Examples

```
rotate_array([1,2,3,4,5,6,7], 3) -> [5,6,7,1,2,3,4]
rotate_array([-1,-100,3,99], 2) -> [3,99,-1,-100]
rotate_array([1,2], 0) -> [1,2]
```
