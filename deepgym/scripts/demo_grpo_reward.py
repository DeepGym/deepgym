"""Demo: Using DeepGym as reward function for GRPO-style training.

This shows how DeepGym reward signals work with group-based RL algorithms
like GRPO, DAPO, Dr.GRPO, REINFORCE++, etc.

The key insight: GRPO needs MULTIPLE samples per prompt, scored independently.
DeepGym's run_batch() provides exactly this — parallel execution with ordered results.

Run:
    python scripts/demo_grpo_reward.py
"""

from pathlib import Path

from deepgym import DeepGym, Environment
from deepgym.integrations.reward import RewardFunction


def main() -> None:
    """Demonstrate GRPO-style reward computation with DeepGym."""
    # ------------------------------------------------------------------
    # 1. Set up environment (python sorting task)
    # ------------------------------------------------------------------
    examples_dir = Path(__file__).resolve().parent.parent / 'examples' / 'python_sorting'
    env = Environment(
        task=examples_dir.joinpath('task.md').read_text(encoding='utf-8'),
        verifier_code=examples_dir.joinpath('verifier.py').read_text(encoding='utf-8'),
        language='python',
        timeout=30,
        difficulty='easy',
        domain='coding',
        tags=['sorting', 'algorithms'],
    )

    # ------------------------------------------------------------------
    # 2. Create reward function
    # ------------------------------------------------------------------
    dg = DeepGym(mode='local')
    reward_fn = RewardFunction(env=env, dg=dg, max_parallel=8)

    # ------------------------------------------------------------------
    # 3. Simulate GRPO: N candidate solutions per prompt
    #    In real training, these come from the policy model.
    # ------------------------------------------------------------------
    candidates = [
        # Correct merge sort
        '''
def sort_list(lst):
    if len(lst) <= 1:
        return lst[:]
    mid = len(lst) // 2
    left = sort_list(lst[:mid])
    right = sort_list(lst[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
''',
        # Correct but uses built-in (may lose partial credit)
        '''
def sort_list(lst):
    return sorted(lst)
''',
        # Partially correct — off by one for some cases
        '''
def sort_list(lst):
    if len(lst) <= 1:
        return lst[:]
    for i in range(len(lst)):
        for j in range(i + 1, len(lst)):
            if lst[i] > lst[j]:
                lst[i], lst[j] = lst[j], lst[i]
    return lst
''',
        # Wrong — returns reversed list
        '''
def sort_list(lst):
    return list(reversed(lst))
''',
        # Crashes — missing function
        '''
def my_sort(lst):
    return sorted(lst)
''',
        # Empty solution
        '''
pass
''',
    ]

    print('=' * 78)
    print('DeepGym GRPO Reward Demo')
    print('=' * 78)
    print()

    # ------------------------------------------------------------------
    # 4. Score all candidates with DeepGym
    # ------------------------------------------------------------------
    scores = reward_fn(candidates)

    # ------------------------------------------------------------------
    # 5. Compute GRPO-style group advantages
    #    advantage_i = (r_i - mean(G)) / std(G)
    # ------------------------------------------------------------------
    group_mean = sum(scores) / len(scores)
    variance = sum((s - group_mean) ** 2 for s in scores) / len(scores)
    group_std = max(variance ** 0.5, 1e-8)  # avoid division by zero

    advantages = [(s - group_mean) / group_std for s in scores]

    # ------------------------------------------------------------------
    # 6. Print results table
    # ------------------------------------------------------------------
    labels = [
        'Correct merge sort',
        'Uses built-in sorted()',
        'Bubble sort (in-place)',
        'Wrong: reversed list',
        'Crash: missing fn name',
        'Empty solution',
    ]

    header = f'{"Solution":<25} {"Score":>6} {"Mean":>6} {"Std":>6} {"Advantage":>10}'
    print(header)
    print('-' * len(header))

    for label, score, adv in zip(labels, scores, advantages):
        print(
            f'{label:<25} {score:>6.3f} {group_mean:>6.3f} {group_std:>6.3f} {adv:>+10.3f}'
        )

    print()
    print('How GRPO uses these advantages:')
    print('  - Positive advantage -> reinforce this solution (increase probability)')
    print('  - Negative advantage -> suppress this solution (decrease probability)')
    print('  - The training framework (TRL, OpenRLHF, verl) handles the policy')
    print('    gradient update; DeepGym just provides the reward signal.')

    # ------------------------------------------------------------------
    # 7. Show shaped rewards (multi-dimensional)
    # ------------------------------------------------------------------
    print()
    print('Shaped reward components (when verifier provides them):')
    print('-' * 50)

    shaped = reward_fn.shaped_rewards(candidates[:3])
    for label, components in zip(labels[:3], shaped):
        print(f'  {label}: {components}')

    print()
    print('Shaped rewards enable fine-grained credit assignment in algorithms')
    print('like Dr.GRPO that can leverage multi-dimensional reward signals.')


if __name__ == '__main__':
    main()
