def read_markdown_file(file_path: str) -> str:
    """
    Read the contents of a markdown file and return it as a string.

    This function opens the specified markdown file, reads its entire content,
    and returns it as a string. If the file is not found or cannot be read,
    an appropriate error message is printed and an empty string is returned.

    Args:
        file_path (str): The path to the markdown file to be read.

    Returns:
        str: The content of the markdown file as a string.
             Returns an empty string if the file cannot be read.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If there's an issue reading the file (e.g., permission error).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return ""
    except IOError:
        print(f"Error: Unable to read file at {file_path}")
        return ""