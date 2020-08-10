import sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5 import QtWidgets, QtGui

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

import numpy as np
from scipy.interpolate import splrep, splev

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


class DraggablePoint:

    # http://stackoverflow.com/questions/21654008/matplotlib-drag-overlapping-points-interactively

    lock = None #  only one can be animated at a time

    def __init__(self, parent, ax, name=None, x=0.1, y=0.1, sizex=0.01,sizey=0.03,color='r', fix_x=True):

        self.ax = ax
        if name is not None:
            self.line = parent.lines[name]
            self.list_points = parent.points[name]
            self.scale = parent.scale[name]
            self.ylim = parent.ylim[name]
        else:
            self.line = parent.line
            self.list_points = parent.list_points
            self.ylim = (0,1)
            self.scale = "linear"

        self.fix_x = fix_x

        self.parent = parent
        self.point = patches.Ellipse((x, y), sizex, sizey, fc=color, alpha=0.5, edgecolor=color)
        self.x = x
        self.y = y

        self.ax.add_patch(self.point)
        #parent.axs["Vd"].add_patch(self.point)
        self.press = None
        self.background = None
        self.connect()


    def connect(self):

        'connect to all the events we need'

        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)


    def on_press(self, event):

        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self

        # draw everything but the selected rectangle and store the pixel buffer
        canvas = self.point.figure.canvas
        ax = self.point.axes
        self.point.set_animated(True)

        self.line.set_animated(True)


        ##canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)

        # now redraw just the rectangle
        ax.draw_artist(self.point)

        # and blit just the redrawn area
        canvas.blit(ax.bbox)


    def on_motion(self, event):

        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        #self.point.height *= 2

        dx = 0 if self.fix_x else event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0]+dx, self.point.center[1]+dy)

        canvas = self.point.figure.canvas
        ax = self.point.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current rectangle
        ax.draw_artist(self.point)

        self.x = self.point.center[0]
        self.y = self.point.center[1]


        ax.draw_artist(self.line)
        points_x = [p.x for p in self.list_points]
        points_y = [p.y for p in self.list_points]
        spl = splrep(points_x, points_y, k=3)
        line_y = splev(self.parent.line_x, spl, ext=3)
        self.line.set_ydata(line_y)

        # blit just the redrawn area
        canvas.blit(ax.bbox)


    def on_release(self, event):

        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)

        self.line.set_animated(False)

        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

        self.x = self.point.center[0]
        self.y = self.point.center[1]

    def disconnect(self):

        'disconnect all the stored connection ids'

        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)

class SingleProfile(FigureCanvas):

    """A canvas that updates itself every second with a new plot."""

    def __init__(self, parent=None, width=8, height=5, dpi=100, fix_x=True, nPoint=11, nPointLine=101):

        self.fix_x = fix_x
        self.nPoint = nPoint
        self.nPointLine = nPointLine

        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.axes.grid(True)

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # To store the 2 draggable points
        self.list_points = []


        self.show()
        self.plotDraggablePoints()


    def plotDraggablePoints(self, size=0.03):

        """Plot and define the 2 draggable points of the baseline"""

        # del(self.list_points[:])

        # define points
        x0, x1 = 0, 1
        fix_x = self.fix_x
        points_x = np.linspace(x0, x1, self.nPoint, endpoint=True)
        points_y = 0.5 * np.ones(points_x.shape)

        # plot line
        self.line_x = np.linspace(x0, x1, self.nPointLine, endpoint=True)
        spl = splrep(points_x, points_y, k=3)
        line_y = splev(self.line_x, spl, ext=3)
        self.line = Line2D(self.line_x, line_y, color='r', alpha=0.5)
        self.fig.axes[0].add_line(self.line)

        # plot points
        for x, y in zip(points_x, points_y):
            self.list_points.append(DraggablePoint(self, self.axes, x=x, y=y, sizex=size,sizey=3*size, fix_x=fix_x))


        self.updateFigure()


    def clearFigure(self):

        """Clear the graph"""

        self.axes.clear()
        self.axes.grid(True)
        del(self.list_points[:])
        self.updateFigure()


    def updateFigure(self):

        """Update the graph. Necessary, to call after each plot"""

        self.draw()

def real_to_fake(y0, ylim_real, ylim_fake=(0,1)):
    _r = (ylim_real[1] - ylim_real[0]) / (ylim_fake[1] - ylim_fake[0])
    return ylim_fake[0] + (y0 - ylim_real[0]) / _r

def fake_to_real(y0, ylim_real, ylim_fake=(0,1)):
    _r = (ylim_real[1] - ylim_real[0]) / (ylim_fake[1] - ylim_fake[0])
    return ylim_real[0] + (y0 - ylim_fake[0]) * _r

class InteractiveProfiles(QtWidgets.QWidget):


    def __init__(self, parent=None, width=8, height=5, dpi=50, p=None, fix_x=True, nPoint=11, nPointLine=101):
        super().__init__()

        self.fix_x = fix_x
        self.nPoint = nPoint
        self.nPointLine = nPointLine
        self.p = p

        self.initUI(width, height)

        self.initFigure(dpi)

        if p is not None:
            self.initAxe(p)
            self.initPlot()

            self.initButton()

        self.ResultWidget =InteractiveResult(parent=self)

        self.show()


    def initUI(self, width, height):
        # widget of Figure
        self.FigureWidget = QtWidgets.QWidget(self)
        # add layout to FigureWidget
        self.FigureLayout = QtWidgets.QVBoxLayout(self.FigureWidget)
        # remove Margin
        self.FigureLayout.setContentsMargins(0,0,0,0)
        # layout
        #self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        w, h = width*100, height*100
        self.setGeometry(100,100,w,h)
        self.setFixedSize(w, h)

        self.FigureWidget.setGeometry(0,0,w,h)
        #self.FigureWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)




    def initFigure(self, dpi=50):
        self.fig = plt.figure(dpi=dpi)
        self.figCanvas = FigureCanvas(self.fig)
        self.FigureLayout.addWidget(self.figCanvas, stretch=8)

        self.ax = self.fig.add_subplot(1,1,1)
        #self.ax.grid(True)
        self.axs = {}

    def initButton(self):
        self.Buttons = InteractiveProfilesButtons(self, self.FigureWidget, self.FigureLayout)



    def initAxe(self, p):

        nAxis = len(p)
        if nAxis > 2:
            pad = 0.9 - (nAxis-2) * 0.09
            self.fig.subplots_adjust(right=pad)

        names = list(p.keys())
        self.names = names
        self.ylim = {}
        self.scale = {}
        for i, name in enumerate(names):
            pv = p[name]

            if i == 0:
                self.axs[name] = self.ax
            else:
                self.axs[name] = self.ax.twinx()
            ax = self.axs[name]

            if i > 1:
                ax.spines["right"].set_position(( "axes", 1+(i-1)*0.15 ))
                make_patch_spines_invisible(ax)
                ax.spines["right"].set_visible(True)

            color = pv[4]

            ax.set_yscale(pv[0])
            self.scale[name] = pv[0]
            if pv[0] == "log":
                self.ylim[name] = tuple( np.log10(pv[1]) )
                ylim = pv[1]
            else:
                ylim = pv[1]
                self.ylim[name] = ylim
            ylabel = name + pv[3]
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylabel, color=color)
            ##ax.yaxis.set_label_coords(-0.1,1.02)
            ax.tick_params(axis='y', colors=color)

    def initPlot(self):

        fix_x = self.fix_x
        names = self.names
        self.lines = {}
        self.points = {}

        # define points
        x0, x1 = 0, 1
        y0, y1 = 0, 1
        self.ylim_fake = (y0, y1)
        points_x = np.linspace(x0, x1, self.nPoint, endpoint=True)
        self.line_x = np.linspace(x0, x1, self.nPointLine, endpoint=True)

        sizex = 0.02
        ax = self.ax.twinx()
        ax.set_axis_off()
        ax.set_ylim(y0, y1)
        for name in names:

            self.points[name] = []
            color = self.p[name][-1]
            scale = self.scale[name]
            ylim = self.ylim[name]

            yi = self.p[name][2]
            if scale == "log":
                yi = np.log10(yi)
            yi = real_to_fake(yi, ylim, ax.get_ylim())

            # plot line
            points_y = yi * np.ones(points_x.shape)
            spl = splrep(points_x, points_y, k=3)
            line_y = splev(self.line_x, spl, ext=3)
            line = Line2D(self.line_x, line_y, color=color, alpha=0.5, label=name)
            self.lines[name] = line
            ax.add_line(line)

            # plot points
            sizey = sizex * 3
            for x, y in zip(points_x, points_y):
                self.points[name].append(DraggablePoint(self, ax, name, x, y, sizex=sizex, sizey=sizey, color=color, fix_x=fix_x))
        ax.legend(loc="upper right", bbox_to_anchor=(0.9, 1., 0.1, 0.1))

class InteractiveProfilesButtons(QtWidgets.QWidget):

    def __init__(self, parent, parentWidget, parentLayout):
        super().__init__()

        self.parent = parent
        self.lines = parent.lines
        self.points = parent.points
        self.names = parent.names
        self.scale = parent.scale
        self.ylim = parent.ylim

        self.Btn_result = QtWidgets.QPushButton("Calculate", parentWidget)
        self.BtnLayout = QtWidgets.QHBoxLayout()
        self.BtnLayout.setContentsMargins(0,0,0,0)
        self.BtnLayout.addWidget(self.Btn_result)
        parentLayout.addLayout(self.BtnLayout, stretch=1)

        self.Btn_result.clicked.connect( self.on_click_result )


    def on_click_result(self):

        #line_x = self.parent.line_x
        #name = "Te"
        #ylim_real = self.ylim[name]
        #points = self.points[name]
        #line = self.lines[name]
        #ylim_fake = self.parent.ylim_fake

        #line_y = fake_to_real(line.get_ydata(), ylim_real, ylim_fake)

        self.parent.ResultWidget.update_line()

class InteractiveResult(QtWidgets.QWidget):


    def __init__(self, parent=None, width=5, height=3, dpi=50):
        super().__init__()

        self.parent = parent

        self.initUI(width, height)

        self.initFigure(dpi)
        self.initAxe()
        self.initPlot()

        self.show()

    def initUI(self, width, height):
        # widget of Figure
        self.FigureWidget = QtWidgets.QWidget(self)
        # add layout to FigureWidget
        self.FigureLayout = QtWidgets.QVBoxLayout(self.FigureWidget)
        # remove Margin
        self.FigureLayout.setContentsMargins(0,0,0,0)
        # layout
        #self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)
        w, h = width*100, height*100
        self.setGeometry(100,100,w,h)
        self.setFixedSize(w, h)

        self.FigureWidget.setGeometry(0,0,w,h)
        #self.FigureWidget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,QtWidgets.QSizePolicy.Expanding)




    def initFigure(self, dpi=50):
        self.fig = plt.figure(dpi=dpi)
        self.figCanvas = FigureCanvas(self.fig)
        self.FigureLayout.addWidget(self.figCanvas, stretch=8)

        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.grid(True)

    def initAxe(self):

        pass

    def initPlot(self):

        line_x = self.parent.line_x
        name = "Te"
        ylim_real = self.parent.ylim[name]
        points = self.parent.points[name]
        line = self.parent.lines[name]
        ylim_fake = self.parent.ylim_fake
        line_y = fake_to_real(line.get_ydata(), ylim_real, ylim_fake)


        self.line, = self.ax.plot(line_x, line_y, color='k', alpha=0.5)
        self.ax.set_xlim(line_x[0],line_x[-1])
        self.ax.set_ylim(ylim_real)
        #self.fig.canvas.draw()

    def update_line(self):

        line_x = self.parent.line_x
        name = "Te"
        ylim_real = self.parent.ylim[name]
        #points = self.parent.points[name]
        line = self.parent.lines[name]
        ylim_fake = self.parent.ylim_fake
        line_y = fake_to_real(line.get_ydata(), ylim_real, ylim_fake)

        self.line.set_animated(True)
        self.line.set_ydata(line_y)
        self.line.set_animated(False)
        self.fig.canvas.draw()







def start_app(app, p):
    r"""
    """
    QApp = QtWidgets.QApplication(sys.argv)
    ex = app(p=p)
    ex.show()
    sys.exit(QApp.exec_())


if __name__ == '__main__':

    p = {
        "Te" : ["linear", (3E3, 2E4), 1E4, "[$K$]", "r"],
        "Ne" : ["log", (1E10,1E12), 5E11, "[$cm^{-3}$]", "k"],
        "Vd" : ["linear", (-6, 6), 0, "[$km/s$]", "b"],
    }

    QApp = QtWidgets.QApplication(sys.argv)
    ex = InteractiveProfiles(p=p)
    #ex = SingleProfile()
    sys.exit(QApp.exec_())
