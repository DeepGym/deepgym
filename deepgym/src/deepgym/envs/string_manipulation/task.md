# String Manipulation

Write a Python function `transform(s: str) -> str` that reverses each word in a string while keeping the original word order.

## Requirements

- Words are separated by single spaces.
- Each word is reversed individually; the order of words stays the same.
- Preserve leading/trailing spaces and multiple consecutive spaces as-is.
- An empty string should return an empty string.

## Examples

```python
transform("hello world")          # -> "olleh dlrow"
transform("Python is fun")        # -> "nohtyP si nuf"
transform("")                     # -> ""
transform("a")                    # -> "a"
transform("  hello  world  ")     # -> "  olleh  dlrow  "
```

## Signature

```python
def transform(s: str) -> str:
    ...
```
