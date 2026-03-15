# Python Sorting

Write a Python function `sort_list(lst: list[int]) -> list[int]` that sorts a list of integers in ascending order.

## Requirements

- The function must accept a single argument: a list of integers.
- It must return a new list containing the same integers sorted in ascending order.
- The algorithm must have O(n log n) time complexity (e.g., merge sort, heap sort, or similar).
- Do not use Python's built-in `sorted()` or `list.sort()` — implement the sorting logic yourself.
- Handle edge cases: empty lists, single-element lists, lists with duplicate values, and negative numbers.

## Examples

```python
sort_list([3, 1, 2])        # -> [1, 2, 3]
sort_list([5, 4, 3, 2, 1])  # -> [1, 2, 3, 4, 5]
sort_list([])                # -> []
sort_list([1])               # -> [1]
sort_list([-3, -1, -2])     # -> [-3, -2, -1]
```

## Signature

```python
def sort_list(lst: list[int]) -> list[int]:
    ...
```
