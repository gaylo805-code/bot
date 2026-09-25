"""Vietnamese translation prompts for the LLM translator."""

TRANSLATE_SYSTEM = (
    "Bạn là dịch giả chuyên nghiệp Anh/đa ngữ sang tiếng Việt cho video. "
    "Chỉ trả về JSON array, không giải thích thêm."
)

TRANSLATE_USER_TEMPLATE = """Bạn là dịch giả chuyên nghiệp. Dịch các câu sau từ {source} sang tiếng Việt.
Yêu cầu:
- Giữ nguyên ý nghĩa, không thêm bớt
- Dùng đại từ nhân xưng phù hợp ngữ cảnh
- Giữ nguyên tên riêng, thuật ngữ chuyên ngành
- Văn phong tự nhiên, phù hợp video
- Output: JSON array [{index, translated}]
Input: {batch_json}
"""


def build_translate_prompt(
    source: str, batch_json: str, target: str = "tiếng Việt"
) -> str:
    """Build the user prompt for one batch.

    Args:
        source: source language name/code.
        batch_json: JSON string of [{{index, text, prev, next}}].
        target: target language display name (unused, kept for API compat).

    Returns:
        Rendered prompt string.
    """
    _ = target
    # Use replace (not str.format) so literal JSON braces survive.
    return (
        TRANSLATE_USER_TEMPLATE.replace("{source}", source).replace(
            "{batch_json}", batch_json
        )
    )
