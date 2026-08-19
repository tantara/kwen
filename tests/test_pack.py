"""Chat-template packing unit tests."""

from korean_sft.pack import IM_END, IM_START, apply_chat_template, pack_sft_row


def test_apply_chat_template_has_qwen_markers():
    text = apply_chat_template(
        [
            {"role": "user", "content": "안녕"},
            {"role": "assistant", "content": "안녕하세요."},
        ]
    )
    assert f"{IM_START}user\n안녕{IM_END}" in text
    assert f"{IM_START}assistant\n안녕하세요.{IM_END}" in text


def test_pack_sft_row_puts_answer_in_assistant_span():
    row = pack_sft_row(
        {
            "id": 1,
            "instruction": "써라",
            "answer": "다듬은 글",
            "topic": "카페",
            "environment": "online",
            "register": "casual",
            "age": 22,
            "background": "대학생",
        }
    )
    assert "text" in row
    assistant = row["text"].split(f"{IM_START}assistant\n", 1)[1].split(IM_END, 1)[0]
    assert assistant.strip() == "다듬은 글"
    assert "써라" in row["text"]
    assert f"{IM_START}user" in row["text"]
