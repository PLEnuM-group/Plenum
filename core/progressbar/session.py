# -*- coding: utf-8 -*-

import sys

"""The session module provides global settings for session handling.
"""

# By default SkyLLH will not be in interactive session, i.e. will be in batch
# mode. Hence, progress bars will not be displayed to not screw up the output.
IS_INTERACTIVE_SESSION = True

def enable_interactive_session():
    """Enables interactive session mode.
    """
    global IS_INTERACTIVE_SESSION

    IS_INTERACTIVE_SESSION = True

def disable_interactive_session():
    """Disables interactive session mode.
    """
    global IS_INTERACTIVE_SESSION

    IS_INTERACTIVE_SESSION = False

def is_interactive_session():
    """Checks whether the current session is interactive (True) or not (False).

    Returns
    -------
    check : bool
        True if the current SkyLLH session is interactive, False otherwise.
    """
    return IS_INTERACTIVE_SESSION

def is_python_interpreter_in_interactive_mode():
    """Checks if the Python interpreter is in interactive mode.

    Returns
    -------
    check : bool
        True if the Python interpreter is in interactive mode, False otherwise.
    """
    return bool(getattr(sys, 'ps1', sys.flags.interactive))
