# House Robber

You are a robber planning to rob houses along a street. Each house has a certain amount of money stashed. Adjacent houses have security systems connected — if two adjacent houses are broken into on the same night, the police will be alerted.

Given a list of non-negative integers representing the amount of money at each house, return the maximum amount you can rob without alerting the police.

## Function Signature

```python
def rob(nums: list[int]) -> int:
```

## Parameters
- `nums` — list of non-negative integers representing money at each house

## Returns
- Maximum amount that can be robbed

## Examples

```
rob([1,2,3,1]) -> 4      # rob house 0 and 2
rob([2,7,9,3,1]) -> 12   # rob house 0, 2, and 4
rob([]) -> 0
```
