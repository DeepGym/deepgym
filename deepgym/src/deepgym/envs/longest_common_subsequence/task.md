# Longest Common Subsequence

Given two strings `text1` and `text2`, return the length of their longest common subsequence.

A subsequence is a sequence that can be derived from another sequence by deleting some or no elements without changing the order of the remaining elements.

## Function Signature

```python
def longest_common_subsequence(text1: str, text2: str) -> int:
```

## Parameters
- `text1` — first string
- `text2` — second string

## Returns
- Length of the longest common subsequence

## Examples

```
longest_common_subsequence("abcde", "ace") -> 3   # "ace"
longest_common_subsequence("abc", "abc") -> 3
longest_common_subsequence("abc", "def") -> 0
```
