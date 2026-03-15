"""Reference solution: reverse each word while preserving spacing."""


def transform(s: str) -> str:
    """Reverse each word in a string while keeping word order and spacing.

    Words are contiguous non-space characters. Spaces (including runs of
    multiple spaces and leading/trailing spaces) are preserved exactly.
    """
    result = []
    word = []

    for ch in s:
        if ch == ' ':
            if word:
                result.append(''.join(reversed(word)))
                word = []
            result.append(' ')
        else:
            word.append(ch)

    if word:
        result.append(''.join(reversed(word)))

    return ''.join(result)
