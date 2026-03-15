# Valid Parentheses

Given a string containing just the characters `(`, `)`, `{`, `}`, `[` and `]`, determine if the input string is valid. A string is valid if open brackets are closed by the same type of brackets in the correct order.

## Function Signature

```python
def is_valid(s: str) -> bool:
```

## Parameters
- `s` — a string containing only bracket characters

## Returns
- `True` if the string has valid (balanced) parentheses, `False` otherwise

## Examples

```
is_valid("()") -> True
is_valid("()[]{}") -> True
is_valid("(]") -> False
is_valid("([)]") -> False
is_valid("{[]}") -> True
```
