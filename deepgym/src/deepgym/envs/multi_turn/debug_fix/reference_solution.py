"""Reference multi-step solution for the debug_fix environment.

Demonstrate the expected agent behaviour: read -> edit -> test.
"""

from deepgym.models import Action, Observation


def agent(obs: Observation) -> Action:
    """Solve the debug_fix task in three steps.

    Step 0: Read the buggy file.
    Step 1: Write the fixed code.
    Step 2: Run the tests.
    """
    if obs.step == 0:
        # First step: read the file to understand the bug.
        return Action(
            content='print(open("buggy.py").read())',
            action_type='code',
        )
    if obs.step == 1:
        # Second step: fix the bug (change == 1 to == 0).
        fixed_code = '''\
with open("buggy.py", "w") as f:
    f.write("""\\
def sum_evens(numbers):
    total = 0
    for n in numbers:
        if n % 2 == 0:
            total += n
    return total
""")
print("File fixed")
'''
        return Action(content=fixed_code, action_type='code')
    # Third step: run the tests.
    return Action(
        content=(
            'import subprocess, sys\n'
            'r = subprocess.run('
            '[sys.executable, "test_buggy.py"],'
            ' capture_output=True, text=True)\n'
            'print(r.stdout)\n'
            'if r.returncode == 0: print("DONE")\n'
        ),
        action_type='code',
    )
