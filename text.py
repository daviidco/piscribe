"""Handling of plain-text and Markdown inputs."""


def process_text(local_path):
    """Read a downloaded text or Markdown file and return its contents.

    Args:
        local_path: Path to the downloaded file on disk.

    Returns:
        The full file contents as a UTF-8 string.
    """
    with open(local_path, "r", encoding="utf-8") as f:
        return f.read()
