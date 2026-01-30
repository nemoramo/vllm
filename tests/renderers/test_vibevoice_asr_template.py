# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from jinja2 import Template

from vllm.entrypoints.chat_utils import load_chat_template

from ..utils import VLLM_PATH


def test_vibevoice_asr_chat_template_accepts_openai_audio_part_types():
    template_path = (
        VLLM_PATH / "examples/online_serving/vibevoice_asr/chat_template.jinja"
    )
    assert template_path.exists()

    template_content = load_chat_template(chat_template=template_path)
    template = Template(template_content)

    parts = [
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAAA"}},
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
        # vLLM normalizes OpenAI audio parts to "audio" before rendering.
        {"type": "audio"},
    ]

    for part in parts:
        rendered = template.render(
            messages=[
                {
                    "role": "user",
                    "content": [
                        part,
                        {"type": "text", "text": "Transcribe."},
                    ],
                }
            ],
            add_generation_prompt=False,
        )
        assert rendered.count("<|AUDIO|>") == 1
        assert "<##AUDIO##>" not in rendered
