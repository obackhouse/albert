"""Printing tools.
"""

from collections import OrderedDict
import re


ANSI_RESET = "\033[0m"

ANSI_COLOURS = OrderedDict([
    ("red", "\033[91m"),
    ("green", "\033[92m"),
    ("yellow", "\033[93m"),
    ("blue", "\033[94m"),
    ("magenta", "\033[95m"),
    ("cyan", "\033[96m"),
])


def highlight_string(string, marker, colour="red", regex=False):
    """Highlight a substring in a string.

    Parameters
    ----------
    string : str
        The string to print.
    marker : str
        The substring to highlight.
    colour : str, optional
        The colour to highlight the substring. Default value is `"red"`.
    regex : bool, optional
        Whether to use the marker as a regular expression. Default
        value is `False`.

    Returns
    -------
    string : str
        The string with the highlighted substring.
    """
    if regex:
        matches = re.findall(marker, string)
        for match in matches:
            string = string.replace(match, f"{ANSI_COLOURS[colour]}{match}{ANSI_RESET}")
    else:
        string = string.replace(marker, f"{ANSI_COLOURS[colour]}{marker}{ANSI_RESET}")
    return string

