"""Buggy module: compute the sum of even numbers in a list."""


def sum_evens(numbers):
    """Return the sum of all even numbers in the list."""
    total = 0
    for n in numbers:
        if n % 2 == 1:  # Bug: should be == 0
            total += n
    return total
