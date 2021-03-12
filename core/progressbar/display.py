# -*- coding: utf-8 -*-

"""The display module provides global settings for pretty command line
displaying.
"""

# Define the width (number of characters) of the display.
PAGE_WIDTH = 80

# Define the width (number of characters) for each text block indentation.
INDENTATION_WIDTH = 4


class ANSIColors:
    """This class defines the ANSI color codes, which can be used to change
    the text color in a terminal.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def add_leading_text_line_padding(padwidth, text):
    """Adds leading white spaces to all the lines of the given text.

    Parameters
    ----------
    padwidth : int
        The width of the padding.
    text : str
        The text with new line characters for each line.

    Returns
    -------
    padded_text : str
        The text where each line is padded with the given number of whitespaces.
    """
    return '\n'.join([ ' '*padwidth + line for line in text.split('\n') ])
