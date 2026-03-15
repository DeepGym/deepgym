# Merge Intervals

Given an array of intervals where `intervals[i] = [start_i, end_i]`, merge all overlapping intervals and return an array of the non-overlapping intervals that cover all the intervals in the input.

## Function Signature

```python
def merge(intervals: list[list[int]]) -> list[list[int]]:
```

## Parameters
- `intervals` — list of [start, end] pairs

## Returns
- List of merged non-overlapping intervals, sorted by start time

## Examples

```
merge([[1,3],[2,6],[8,10],[15,18]]) -> [[1,6],[8,10],[15,18]]
merge([[1,4],[4,5]]) -> [[1,5]]
merge([]) -> []
```
