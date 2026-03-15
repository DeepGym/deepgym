"""DeepGym Demo -- HuggingFace Spaces app.

Try DeepGym environments in your browser.
Run with: gradio app.py
"""

from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)

from deepgym import DeepGym, list_environments, load_environment

dg = DeepGym(mode='local')


def get_env_names() -> list[str]:
    """Return sorted list of available environment names."""
    return sorted(e['name'] for e in list_environments())


def get_task(env_name: str) -> str:
    """Load and return the task description for an environment.

    Args:
        env_name: Name of the environment.

    Returns:
        Task description as markdown string.
    """
    if not env_name:
        return ''
    env = load_environment(env_name)
    return env.task


def run_code(env_name: str, code: str) -> str:
    """Run user code against the selected environment verifier.

    Args:
        env_name: Name of the environment to run against.
        code: User-provided solution source code.

    Returns:
        Formatted result string with score, pass/fail, and details.
    """
    if not env_name:
        return 'Select an environment first.'
    if not code.strip():
        return 'Write some code first.'
    try:
        env = load_environment(env_name)
        result = dg.run(env, model_output=code)
        lines = [
            f'Score: {result.score}',
            f'Passed: {result.passed}',
        ]
        if result.output:
            lines.append(f'Details: {result.output}')
        return '\n'.join(lines)
    except Exception as exc:
        logger.error('run_code failed', exc_info=True)
        return f'Error: {exc}'


env_names = get_env_names()

with gr.Blocks(title='DeepGym') as demo:
    gr.Markdown(
        '# DeepGym -- RL Training Environments\n'
        'Test your code against DeepGym verifiers. '
        'Used for GRPO/RL training of coding agents.'
    )

    with gr.Row():
        env_dropdown = gr.Dropdown(
            choices=env_names,
            label='Environment',
            interactive=True,
        )

    task_display = gr.Markdown(label='Task')
    code_input = gr.Code(language='python', label='Your Solution')
    run_btn = gr.Button('Run', variant='primary')
    result_output = gr.Textbox(label='Result', lines=6)

    env_dropdown.change(fn=get_task, inputs=env_dropdown, outputs=task_display)
    run_btn.click(fn=run_code, inputs=[env_dropdown, code_input], outputs=result_output)

demo.launch()
