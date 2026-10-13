# Built-in environments

This page lists the 31 environments that ship with the package and shows how to load
them, run them as suites, and read their source.

```python
from deepgym import list_environments, load_environment

env = load_environment('coin_change')
for entry in list_environments():
    print(entry['path'], entry.get('type', 'coding'), entry['difficulty'])
```

## Coding (22)

| Family | Environments |
|---|---|
| Array and string | `reverse_string`, `palindrome_check`, `anagram_check`, `max_subarray`, `rotate_array`, `remove_duplicates`, `valid_parentheses` |
| Hash map | `group_anagrams`, `longest_consecutive`, `top_k_frequent` |
| Dynamic programming | `climbing_stairs`, `coin_change`, `longest_common_subsequence`, `house_robber` |
| Graph and tree | `binary_search`, `merge_intervals`, `level_order_traversal` |
| Practical | `fizzbuzz`, `roman_to_integer`, `matrix_spiral` |
| Filesystem and CLI scripting | `file_organizer`, `cli_task` |

## Computer use (5)

Multi-turn desktop tasks on Daytona, each with 8 seeded variants: `signup_form`,
`todo_list`, `settings_tabs`, `expense_sheet`, `file_manager`. See
[computer use](computer-use.md).

## Tool use (2)

`api_request`, `data_pipeline`.

## Cyber (2)

`cyber/cve_2021_44228_log_triage` (Log4Shell log triage) and
`cyber/web_path_traversal_toy` (a synthetic CTF-style path traversal). Both are
local-only; see [CyberBench](../cyberbench.md) for the rules new cyber tasks follow.

## Suites

`load_suite(name)` and `deepgym eval --suite NAME` take a difficulty (`easy`, `medium`,
`hard`), a type (`coding`, `computer-use`, `tool-use`), `all`, or a family name such as
`dynamic-programming`. Computer-use environments need an agent, so they only appear in
the `computer-use` suite.

```bash
# one solution file per environment, named after it: solutions/coin_change.py, ...
deepgym eval --suite dynamic-programming --solutions-dir solutions/
```

## Where they live

Each built-in is a folder in `src/deepgym/envs/` with `task.md`, `verifier.py`,
`reference_solution.py` and `metadata.json`, listed in `src/deepgym/envs/registry.json`.
Computer-use tasks add `setup.sh`, `variants.json`, an `app/` folder and a
`reference_actions.json` with one scripted episode per variant.

The test suite runs every single-turn built-in through the
[QA gate](../authoring/qa-gate.md): the reference scores 1.0 on every run, and empty, stub
and reward-hack answers earn nothing. The computer-use built-ins are checked with
`deepgym qa --mode daytona`, since they need a desktop. To add an environment, see
[authoring](../authoring/new-check-qa.md).
