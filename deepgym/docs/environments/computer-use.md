# Computer use

This page covers desktop environments: how an episode runs, the actions an agent can
take, seeded variants, how verifiers score the result, and how the desktop is locked
down. Computer-use environments need Daytona (`pip install "deepgym[daytona]"` and
`DAYTONA_API_KEY`).

## An episode

A computer-use environment boots Daytona's desktop (xfce and Chromium), runs a setup
script, and hands the agent a screenshot. Actions are clicks, typing, keys, scrolls and
drags in screenshot pixels. The verifier runs in the same sandbox and checks what the
agent changed (saved app state, files), never pixels.

```python
from deepgym import Action, DeepGym, load_environment

dg = DeepGym(mode='daytona')
env = load_environment('settings_tabs')

with dg.session(env) as session:
    obs = session.start()                     # obs.screenshot is a base64 PNG
    while not session.done:
        action = my_agent(obs)                # e.g. Action(action_type='click', x=376, y=121)
        obs, reward, done = session.step(action)
    result = session.score()
print(result.score, result.output)            # "3/3 changes made"
```

Or in one call: `trajectory, result = dg.run_episode(env, my_agent)`. To keep the episode
as training data, record it with [`dg.trace()`](../traces.md).

## Actions

| `action_type` | Arguments |
|---|---|
| `click`, `double_click`, `right_click`, `move` | `x`, `y` |
| `drag` | `x`, `y`, `to_x`, `to_y` |
| `scroll` | `x`, `y`, `content` (`up`, `down`, `left` or `right`), `amount` |
| `type` | `content`: the text |
| `key` | `content`: a key chord such as `ctrl+s` |
| `screenshot`, `wait` | `wait` takes `amount` in seconds |
| `done` | Ends the episode. |

Computer-use environments allow only these GUI actions by default, so the agent can't
skip the interface with a shell. An action the environment doesn't allow, or one the
desktop rejects (an unknown key name, say), raises `ActionError`.

## Tasks and variants

The built-in desktop tasks are `signup_form`, `todo_list`, `settings_tabs`,
`expense_sheet` and `file_manager`. Each has 8 seeded variants (different names, values,
files) so a policy can't memorize one answer: `load_environment('todo_list', variant=3)`.
The count is in `list_environments()` under `variants`.

## Scoring

Verifiers split checks into *progress* (the changes the task asked for) and *guards*
(work was saved, nothing else broke, no duplicate rows). The score is the share of
progress checks passed, and any failed guard scores 0, so neither doing nothing nor
clicking Save on untouched state earns reward.

Each task ships a `reference_actions.json` with one scripted episode per variant. On
`--mode daytona`, `deepgym qa` replays it and runs agents that quit at once, use every
action or act at random, each of which must earn nothing. See the
[QA gate](../authoring/qa-gate.md).

## Isolation and lockdown

- The desktop session and every app it starts run as an unprivileged user with no
  network and no sudo, separate from the verifier's user, so anything the agent starts
  from the GUI can't reach root, the internet or the score. Only Xvfb, x11vnc and
  Daytona's input relay stay with Daytona's sudoer; none of them runs commands for X
  clients. Setup scripts don't get sudo either.
- Before setup runs, the lockdown removes the known ways out of the GUI: terminals, run
  dialogs, browser devtools, Thunar custom actions and "Open With" launches (custom
  commands included). That list can't be complete, so it is defense in depth; the
  boundaries above don't depend on it. Chromium runs under a root-owned managed policy
  that turns DevTools off and blocks `chrome://`, `file://` and similar URLs.
- Every lockdown step is checked once applied and again once the desktop runs. A missing
  helper, a desktop already running, or a session that would start as Daytona's sudoer
  raises `SandboxError` instead of handing the agent an open desktop.

## Training

verl's multi-turn agent loop runs computer-use episodes with one sandbox per rollout and
returns screenshots to the model as images. See [verl](../training/verl.md#multi-turn-agents).
