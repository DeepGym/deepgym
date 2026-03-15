# Binary Tree Level Order Traversal

Given a binary tree represented as a list (where index `i` has children at `2*i+1` and `2*i+2`, and `None` represents missing nodes), return its level order traversal as a list of lists.

## Function Signature

```python
def level_order(tree: list) -> list[list[int]]:
```

## Parameters
- `tree` — a list representing a binary tree in BFS order. `None` values represent missing nodes.

## Returns
- A list of lists, where each inner list contains the values at that level from left to right.

## Examples

```
level_order([3,9,20,None,None,15,7]) -> [[3],[9,20],[15,7]]
level_order([1]) -> [[1]]
level_order([]) -> []
```
