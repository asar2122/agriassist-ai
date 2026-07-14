import re


def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def validate_number(value, minimum, maximum):

    try:

        value = float(value)

        return minimum <= value <= maximum

    except (ValueError, TypeError):

        return False