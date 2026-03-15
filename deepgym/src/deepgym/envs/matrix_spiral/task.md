# Spiral Matrix

Given an `m x n` matrix, return all elements of the matrix in spiral order.

## Function Signature

```python
def spiral_order(matrix: list[list[int]]) -> list[int]:
```

## Parameters
- `matrix` — an m x n 2D list of integers

## Returns
- A list of integers in spiral order (clockwise from top-left)

## Examples

```
spiral_order([[1,2,3],[4,5,6],[7,8,9]]) -> [1,2,3,6,9,8,7,4,5]
spiral_order([[1,2,3,4],[5,6,7,8],[9,10,11,12]]) -> [1,2,3,4,8,12,11,10,9,5,6,7]
spiral_order([]) -> []
```
