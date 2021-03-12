# -*- coding: utf-8 -*-
# This is a progressbar stolen from 
# the skyllh framework

from __future__ import division

import numpy as np
import sys
import time

from core.progressbar import display
from core.progressbar import session

def int_cast(v, errmsg, allow_None=False):
    """Casts the given value to an integer value. If the cast is impossible, a
    TypeError is raised with the given error message. If `allow_None` is set to
    `True` the value `v` can also be `None`.
    """
    if(allow_None and v is None):
        return v

    try:
        v = int(v)
    except:
        raise TypeError(errmsg)

    return v



class ProgressBar(object):
    """This class implements a hierarchical progress bar that can serve as a
    parent for child progress bars to display different levels of processing.
    """
    def __init__(self, maxval, startval=0, parent=None):
        """Creates a new ProgressBar instance.

        Parameters
        ----------
        maxval : int
            The maximal value the progress can reach.
        startval : int
            The progress value to start with. Must be smaller than `maxval`.
        parent : instance of ProgressBar
            The parent instance of ProgressBar if this progress bar is a sub
            progress bar.
        """
        super(ProgressBar, self).__init__()

        self.maxval = maxval
        self.startval = startval
        self.parent = parent

        self._start_time = None
        self._val = 0
        self._last_rendered_pbar_str = None
        self._sub_pbar_list = []

    @property
    def maxval(self):
        """The maximal integer value the progress can reach.
        """
        return self._maxval
    @maxval.setter
    def maxval(self, v):
        v = int_cast(v, 'The maxval property must be castable to an integer '
            'value!')
        self._maxval = v

    @property
    def startval(self):
        """The progress integer value to start with. It must be smaller than
        `maxval`.
        """
        return self._startval
    @startval.setter
    def startval(self, v):
        v = int_cast(v, 'The startval property must be castable to an integer '
            ' value!')
        if(v >= self._maxval):
            raise ValueError('The startval value (%d) must be smaller than the '
                'value of the `maxval` property (%d)!',
                v, self._maxval)
        self._startval = v

    @property
    def parent(self):
        """The parent ProgressBar instance of this progress bar, or `None` if
        no parent exist.
        """
        return self._parent
    @parent.setter
    def parent(self, pbar):
        if(pbar is not None):
            if(not isinstance(pbar, ProgressBar)):
                raise TypeError('The parent property must be None, or an '
                    'instance of ProgressBar!')
        self._parent = pbar

    @property
    def progress(self):
        """(read-only) The progress of this progress bar as a number between 0
        and 1.
        """
        return (self._val - self._startval) / (self._maxval - self._startval)

    @property
    def gets_shown(self):
        """(read-only) Flag if the progress bar gets shown. This is ``True``
        if the program is run in an interactive session, ``False`` otherwise.
        """
        return session.is_interactive_session()

    def add_sub_progress_bar(self, pbar):
        """Adds the given progress bar to the list of running sub progress bars
        of this progress bar.
        """
        if(not isinstance(pbar, ProgressBar)):
            raise TypeError('The pbar argument must be an instance of '
                'ProgressBar!')
        self._sub_pbar_list.append(pbar)

    def remove_sub_progress_bar(self, pbar):
        """Removes the given progress bar instance from the list of running sub
        progress bars of this progress bar.
        """
        self._sub_pbar_list.remove(pbar)

    def _sec_to_hms(self, t):
        """Calculate hours, minutes, and seconds from `t` given in seconds.

        Parameters
        ----------
        t : float
            The time in seconds.

        Returns
        -------
        t_h : float
            The integer hours of `t`.
        t_m : float
            The integer minutes of `t`.
        t_s : float
            The integer seconds of `t`.
        """
        t_h = int(t / 3600)
        t -= t_h*3600
        t_m = int(t / 60)
        t -= t_m*60
        t_s = int(np.round(t, 0))

        return (t_h, t_m, t_s)

    def _render_pbar_str(self):
        """Renders the progress bar string. This method is called only when
        this progress bar has no parent.

        Returns
        -------
        pbar_str : str
            The rendered progress bar string.
        """
        # Calculate the elapsed time (ELT) for the first 10 seconds or if we
        # are at the end of the progress. Otherwise calculate the estimated
        # remaining time (ERT).
        curr_time = time.time()
        t_elapsed = curr_time - self._start_time

        t_label = 'ELT'
        t = t_elapsed

        progress = self.progress
        if(progress > 0 and (t_elapsed >= 10) and (self._val < self._maxval)):
            t_label = 'ERT'
            t_total = t_elapsed / progress
            t = t_total - t_elapsed

        (t_h, t_m, t_s) = self._sec_to_hms(t)

        # Get the current progress values from the sub progress bars and this
        # progress bar. The value of this progress bar is the last value.
        p_list = self.get_progress_list()

        sub_pbar_str = ''
        for p in p_list[:-1]:
            if(p != 1):
                sub_pbar_str += '%d '%(int(p*10))
            else:
                sub_pbar_str += '# '

        barwidth = max(display.PAGE_WIDTH - 22 - len(sub_pbar_str), 10)
        nchar = int(np.round(p_list[-1] * barwidth, 0))

        fmt = "%s[%-"+str(barwidth)+"s] %3d%% %s %dh:%02dm:%02ds"
        pbar_str = fmt%(
            sub_pbar_str,
            '='*nchar,
            int(np.round(p_list[-1]*100, 0)),
            t_label, t_h, t_m, t_s
        )

        return pbar_str

    def get_progress_list(self):
        """Retrieves the list of progress values of all the sub progress bars
        and this progress. The value of this progress bar is the last value.
        """
        p_list = []
        for pbar in self._sub_pbar_list:
            p_list.extend(pbar.get_progress_list())
        p_list.append(self.progress)

        return p_list

    def rerender(self):
        """Rerenders this progress bar on the display, but only if the rendered
        progress bar string changed.
        """
        pbar_str = self._render_pbar_str()
        if(pbar_str == self._last_rendered_pbar_str):
            return

        sys.stdout.write('\r'+pbar_str)
        sys.stdout.flush()
        self._last_rendered_pbar_str = pbar_str

    def trigger_rerendering(self):
        """Triggers a rerendering of the most top progress bar.
        """
        parent = self._parent
        if(parent is not None):
            parent.trigger_rerendering()
            return

        if(not session.is_interactive_session()):
            return

        # We are the most top parent progress bar. So we need to get rerendered.
        self.rerender()

    def start(self):
        """Starts the progress bar by taking the start time and setting the
        progress value to the start value.
        If this progress bar has a parent, it adds itself to the parent's list
        of running sub progress bars. Otherwise it will render and print this
        progress bar for the first time.

        Returns
        -------
        self : instance of ProgressBar
            The instance of this ProgressBar.
        """
        self._start_time = time.time()
        self._val = self._startval

        parent = self._parent
        if(parent is not None):
            # Add this progress bar to the parent progress bar.
            parent.add_sub_progress_bar(self)
            return self

        if(not session.is_interactive_session()):
            return self

        self._last_rendered_pbar_str = self._render_pbar_str()

        sys.stdout.write(self._last_rendered_pbar_str)
        sys.stdout.flush()

        return self

    def finish(self):
        """Finishes this progress bar by setting the current progress value to
        its max value.
        If this progress bar has a parent, it triggers a rerendering of the
        parent and then removes itself from the parent's list of running sub
        progress bars. Otherwise it will render and print this progress bar for
        the last time.
        """
        self._val = self._maxval

        parent = self._parent
        if(parent is not None):
            parent.trigger_rerendering()
            # Remove this progress bar from the parent progress bar.
            parent.remove_sub_progress_bar(self)
            return

        if(not session.is_interactive_session()):
            return

        self._last_rendered_pbar_str = self._render_pbar_str()

        sys.stdout.write('\r'+self._last_rendered_pbar_str+"\n")
        sys.stdout.flush()

    def update(self, val):
        """Updates the progress value. If this progress bar has no parent
        progress bar, a rerendering of this progress bar is made.

        Parameters
        ----------
        val : int
            The new current progress value.
        """
        self._val = val

        self.trigger_rerendering()

    def increment(self, dval=1):
        """Updates the progress bar by incrementing the progress by the given
        integral amount.

        Parameters
        ----------
        dval : int
            The amount of progress to increment the progress bar with.
        """
        self.update(self._val + dval)
