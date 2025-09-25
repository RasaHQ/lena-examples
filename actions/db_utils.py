import json
import os
from typing import Any, Union, Text
from pathlib import Path

DEFAULT_ENCODING = "utf-8"


def write_json_to_file(filename: Text, 
                       obj: Any, 
                       **kwargs: Any) -> None:
    """Write an object as a json string to a file."""
    json_string = json_to_string(obj, **kwargs)
    write_to_file(filename, json_string)


def write_to_file(filename: Text, 
                  text: Any) -> None:
    """Write text to a file."""
    write_text_file(str(text), filename)


def json_to_string(obj: Any, 
                   **kwargs: Any) -> Text:
    """Dumps a JSON-serializable object to string with customizable options."""
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)


def write_text_file(content: Text,
                    file_path: Union[Text, Path],
                    encoding: Text = DEFAULT_ENCODING,
                    append: bool = False) -> None:
    """Writes text to a file."""
    mode = 'a' if append else 'w'
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)


def read_json_file(filename: Union[Text, Path]) -> Any:
    """Read json from a file."""
    content = read_file(filename)
    try:
        return json.loads(content)
    except ValueError as e:
        raise ValueError(
            f"Failed to read json from '{os.path.abspath(filename)}'. Error: {e}"
        )


def read_file(filename: Union[Text, Path], 
              encoding: Text = DEFAULT_ENCODING) -> Any:
    """Read text from a file."""
    try:
        with open(filename, encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Failed to read file, " f"'{os.path.abspath(filename)}' does not exist."
        )
    except UnicodeDecodeError:
        raise UnicodeDecodeError(
            f"Failed to read file '{os.path.abspath(filename)}', "
            f"could not read the file using {encoding} to decode "
            f"it. Please make sure the file is stored with this "
            f"encoding."
        )
