import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class MonitorTarget:
    def __init__(self):
        self.radius=7
        self.width=160
        self.height=90
        mpl.rcParams['toolbar'] = 'None'
        plt.ion()
        self.fig = plt.figure()
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        self.fig.canvas.toolbar_visible = False
        self.ax = plt.axes(xlim=(0, self.width), ylim=(0, self.height))
        self.target = plt.Circle((0, 0), self.radius, color='blue')
        self.ax.add_patch(self.target)
        plt.axis('off')

        figManager = plt.get_current_fig_manager()
        figManager.full_screen_toggle()

    def reset_plot(self):
        x, y = np.random.random(2)
        self.target.set_center(
            (self.radius + x * (self.width - 2 * self.radius),
             self.radius + y * (self.height - 2 * self.radius))
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.032)

mt = MonitorTarget()
mt.reset_plot()
mt.reset_plot()
mt.reset_plot()
mt.reset_plot()
mt.reset_plot()

while True:
    pass
