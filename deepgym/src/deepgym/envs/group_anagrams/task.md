# Group Anagrams

Given an array of strings, group the anagrams together. You can return the answer in any order. Each group should be sorted alphabetically, and the groups themselves should be sorted by their first element.

## Function Signature

```python
def group_anagrams(strs: list[str]) -> list[list[str]]:
```

## Parameters
- `strs` — a list of lowercase strings

## Returns
- A list of groups, where each group is a sorted list of anagrams. Groups are sorted by their first element.

## Examples

```
group_anagrams(["eat","tea","tan","ate","nat","bat"])
-> [["ate","eat","tea"], ["bat"], ["nat","tan"]]
```
