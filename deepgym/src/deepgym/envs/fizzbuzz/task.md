# FizzBuzz with Custom Rules

Implement FizzBuzz with custom divisor-word rules. Given `n` and a list of `(divisor, word)` rules, return a list of strings for numbers 1 through n. If a number is divisible by a divisor, append the corresponding word. If no rules match, use the number as a string.

Rules are applied in the order given.

## Function Signature

```python
def fizzbuzz(n: int, rules: list[tuple[int, str]]) -> list[str]:
```

## Parameters
- `n` — positive integer
- `rules` — list of (divisor, word) tuples

## Returns
- List of n strings

## Examples

```
fizzbuzz(5, [(3, "Fizz"), (5, "Buzz")])
-> ["1", "2", "Fizz", "4", "Buzz"]

fizzbuzz(15, [(3, "Fizz"), (5, "Buzz")])
-> ["1","2","Fizz","4","Buzz","Fizz","7","8","Fizz","Buzz","11","Fizz","13","14","FizzBuzz"]
```
