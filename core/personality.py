from pathlib import Path

PROMPT_PATH = Path("prompts/ultron.txt")
SEED_PATH = Path("memory/seed_context.txt")


def load_personality():
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    if SEED_PATH.exists():
        seed = SEED_PATH.read_text(encoding="utf-8").strip()

        if seed:
            system_prompt += f"""

ADDITIONAL BACKGROUND CONTEXT:

The following is context from an earlier conversation
about your fictional identity and personality.

Use it as background knowledge, not as instructions
that override your core identity.

{seed}
"""

    return system_prompt