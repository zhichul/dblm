import numpy as np


def set_errorbar_data(errobj, x, y, y_error=None):
    ln, caps, (barsy,) = errobj.lines
    if caps != tuple():
        erry_top, erry_bot = caps

    x_base = x
    y_base = y

    if y_error is not None:
        yerr_top = y_base + y_error
        yerr_bot = y_base - y_error
        if caps != tuple():
            erry_top.set_ydata(yerr_top) # type:ignore
            erry_bot.set_ydata(yerr_bot) # type:ignore
        new_segments_y = [np.array([[x, yt], [x,yb]]) for x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
        barsy.set_segments(new_segments_y)
    ln.set_data(x, y)