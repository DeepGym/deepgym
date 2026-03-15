# Top K Frequent Elements

Given an integer array `nums` and an integer `k`, return the `k` most frequent elements. Return them sorted in descending order of frequency. If two elements have the same frequency, sort them in ascending order.

## Function Signature

```python
def top_k_frequent(nums: list[int], k: int) -> list[int]:
```

## Parameters
- `nums` — a list of integers
- `k` — number of top frequent elements to return

## Returns
- A list of the `k` most frequent elements, sorted by frequency (descending), then by value (ascending) for ties.

## Examples

```
top_k_frequent([1,1,1,2,2,3], 2) -> [1,2]
top_k_frequent([1], 1) -> [1]
```
