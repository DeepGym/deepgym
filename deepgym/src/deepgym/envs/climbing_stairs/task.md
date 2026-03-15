# Climbing Stairs

You are climbing a staircase. It takes `n` steps to reach the top. Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

## Function Signature

```python
def climb_stairs(n: int) -> int:
```

## Parameters
- `n` — number of steps (1 <= n <= 45)

## Returns
- The number of distinct ways to climb to the top

## Examples

```
climb_stairs(2) -> 2   # (1+1) or (2)
climb_stairs(3) -> 3   # (1+1+1), (1+2), (2+1)
climb_stairs(1) -> 1
```
