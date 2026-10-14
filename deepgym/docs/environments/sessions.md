# Multi-turn sessions

This page covers live multi-turn sessions: one sandbox kept alive for a whole episode,
actions applied step by step, and a final verifier that scores the state the agent left.
It covers the Python API, writing a multi-turn environment, step rewards, and running
sessions over HTTP.

## The session API

```python
from deepgym import Action, DeepGym, MultiTurnEnvironment

env = MultiTurnEnvironment(
    task='Write the word hello into notes.txt, then send done.',
    final_verifier_code=(
        'import json, pathlib\n'
        'p = pathlib.Path("notes.txt")\n'
        'ok = p.is_file() and p.read_text().strip() == "hello"\n'
        'print(json.dumps({"score": float(ok), "passed": ok}))\n'
    ),
)

dg = DeepGym(mode='docker')   # or 'apple', 'modal', 'e2b', 'daytona'; 'local' isn't isolated
with dg.session(env) as session:
    obs = session.start()
    obs, reward, done = session.step(Action(action_type='bash', content='echo hello > notes.txt'))
    obs, reward, done = session.step(Action(action_type='done'))
    result = session.score()
print(result.score, result.has_verdict)
```

- `start()` writes the environment's files, runs its setup, and returns the first
  observation. `step(action)` returns the new observation, the step reward and whether
  the episode is done. `score()` runs the final verifier.
- Or in one call: `trajectory, result = dg.run_episode(env, agent)`, where `agent` maps an
  observation to the next action.
- Shell actions are `code` (Python), `bash` and `write_file` (`path` relative to the work
  directory), plus `done`. Commands run under bash and report `exit_code` in the
  observation's metadata.
- The episode ends on `done`, after `max_steps` (default 10), or when an action's output
  has a line `DONE` (or a JSON line with `"done": true`). Each action has
  `timeout_per_step` seconds (default 30); setup has `setup_timeout` (default 120).
- `trajectory` holds every (observation, action) pair and the step rewards.
- One session runs one episode; open a new one per episode. Concurrent calls on a session
  queue up.

Computer-use environments use the same API with GUI actions; see
[computer use](computer-use.md). To keep an episode as training data, record it with
[`dg.trace()`](../traces.md).

## Backends and isolation

Every provider runs sessions. In a sandbox, agent commands run as an unprivileged
`dgagent` user in a loopback-only network namespace. Verifiers run offline as a separate
`dgverifier` user, with `python -I`, from a file only that user can read, so the agent
can't read, swap or hijack them, and leftover agent processes can't read or forge the
score.

`mode='local'` runs the agent's actions on your machine without isolation; use it only
for agents you trust.

## Writing a multi-turn environment

`deepgym new NAME --kind multi-turn` scaffolds a working one (see
[authoring](../authoring/new-check-qa.md)). The fields:

| Field | What it is |
|---|---|
| `task` | The prompt the agent sees. |
| `files` | Text files written into the work directory before setup. |
| `setup_script`, `setup_code` | Bash, then Python, run after the files are written; the setup output is the first observation. A setup step that exits non-zero raises `SandboxError`. |
| `step_verifier_code` | Optional; runs after every step and returns its reward. |
| `final_verifier_code` | Scores the final state. |
| `max_steps`, `timeout_per_step`, `setup_timeout` | Limits. |
| `allowed_actions` | The action types the agent may use; anything else raises `ActionError`. |
| `image` | The image the sandbox boots from. |

Verifiers print protocol JSON, as single-turn ones do (see
[verifiers](../authoring/verifiers.md)). A score only counts if the verifier exits
cleanly (0, or 1 for a failed run), and a score printed before a crash is never kept. A
verifier that crashes before it opens a file in the agent's work directory or starts the
agent's code has no verdict; after that the crash came from handling the agent's work, so
it scores 0 (`solution_error`). One killed by a signal has no verdict. Only stdout counts,
so a JSON line on stderr can't replace the verdict; stderr is kept in `RunResult.stderr`.

Session verifiers that grade the agent's code load it with `load_solution` too, so it runs
isolated from the verifier: locally in the same sandbox as single runs, and in sandbox
images with sudo (Daytona's default has it) as a third user, `dgsolver`, with its groups
cleared, no-new-privs and an empty capability bounding set. Images without sudo, such as
the default `python:3.12-slim`, have no `deepgym_solution` in sessions.

## Step rewards and verdicts

- A step verifier gets the step's output in the file named by `STEP_OUTPUT_PATH` and the
  step number in `STEP_NUMBER`, and prints a float or protocol JSON. `session.step_verdict`
  holds its last output and exit code.
- A step that fails after its action may have run ends the episode: `step()` raises, the
  action stays on the trajectory with no step reward (so `step_rewards` is one shorter
  than `steps`), and `score()` returns the result that failure fixed instead of running
  the final verifier. A provider or network failure, or a failed step verifier (bad
  output, a timeout, exit 2, or an `error_type` that isn't the model's fault), has no
  verdict; a failed step verifier reports `verifier_error`, or `invalid_score` for a NaN
  or out-of-range score. Anything else follows from the agent's actions (its sandbox
  died, a write failed, its files blocked the step's output) and scores 0 with
  `error_type='policy_sandbox_failure'`.
- A rejected action (not allowed, or the agent's own files in the way of a `write_file`)
  raises `ActionError` and the episode goes on.
- `run_episode` returns the result a failed step fixed instead of raising.
- A sandbox that dies under the final verifier scores 0 (`policy_sandbox_failure`) once the
  agent has acted.
- An environment with no `final_verifier_code` has no verdict: `score()` returns
  `error_type='no_verifier'`, so trainers mask the episode.

See [the reward contract](../training/reward-contract.md).

## Sessions over HTTP

`deepgym serve` exposes sessions to trainers in any language:

```
POST   /v1/sessions                {"environment": "todo_list", "variant": 3}  -> session_id, observation
POST   /v1/sessions/{id}/step      {"action_type": "type", "content": "Pay rent"}
POST   /v1/sessions/{id}/score     -> RunResult
DELETE /v1/sessions/{id}
```

- `environment` can also be an inline `MultiTurnEnvironment` spec; the API refuses one
  that sets an `image` (a `snapshot` is accepted).
- At most 100 sessions run at once (429 above). Sessions idle for 30 minutes are closed
  by a sweep that runs every 60 seconds, and the server tries to close every session on
  shutdown.
- If deleting a session's sandbox fails, `DELETE` answers 502 and the session keeps its
  id and slot, so the delete can be retried; the sweep retries it too, up to 10 attempts.
  After that its sandbox id is logged as an error, the slot is freed, and `/health`
  counts it in `leaked_sandboxes`.

Server setup and the other endpoints are in the [API server](../api-server.md) docs.
