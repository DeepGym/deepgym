# Coin Change

Given an array of coin denominations and a target amount, return the minimum number of coins needed to make that amount. If it's not possible, return -1.

## Function Signature

```python
def coin_change(coins: list[int], amount: int) -> int:
```

## Parameters
- `coins` — list of positive integer denominations
- `amount` — target amount (non-negative integer)

## Returns
- Minimum number of coins needed, or -1 if impossible

## Examples

```
coin_change([1,5,10,25], 30) -> 2   # 25+5
coin_change([2], 3) -> -1
coin_change([1], 0) -> 0
```
