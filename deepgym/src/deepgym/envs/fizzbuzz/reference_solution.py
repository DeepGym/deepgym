def fizzbuzz(n: int, rules: list) -> list:
    result = []
    for i in range(1, n + 1):
        s = ''
        for divisor, word in rules:
            if i % divisor == 0:
                s += word
        result.append(s if s else str(i))
    return result
