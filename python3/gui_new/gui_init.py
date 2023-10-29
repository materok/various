from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import numpy as np

# from data import data
from helper.laufen import data
from helper.gui_utils import MakeKMHPlot

# plot function is created for
# plotting the graph in
# tkinter window
def plot():

    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)

    # list of squares
    y = [i**2 for i in range(101)]

    # adding the subplot
    plot1 = fig.add_subplot(111)

    # plotting the graph
    plot1.plot(y)

    # creating the Tkinter canvas
    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()

def KMHPlot():
    # the figure that will contain the plot
    fig = Figure(figsize = (5, 5),
                 dpi = 100)

    plot1 = fig.add_subplot(111)
    # list of squares
    d=data(2016,2020,True)
    d.drawYears=[2020]
    MakeKMHPlot(plot1, d.run[2020]["bin"],d.run[2020]["velocity"],d.run[2020]["distance"],"")


    canvas = FigureCanvasTkAgg(fig,
                               master = window)
    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().pack()

    # creating the Matplotlib toolbar
    toolbar = NavigationToolbar2Tk(canvas,
                                   window)
    toolbar.update()

    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().pack()

# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("500x500")

# button that displays the plot
plot_button = Button(master = window,
                     command = plot,
                     height = 2,
                     width = 10,
                     text = "Plot")

# place the button
# in main window
plot_button.pack()
quit_button = Button(master=window,text="Quit",height=2,width=10,command=window.destroy)
quit_button.pack()
quit_button.place(x=425, y=0)

kmh_button = Button(master=window,text="kmhplot",height=2,width=10,command=KMHPlot)
kmh_button.pack()
kmh_button.place(x=0, y=0)

# run the gui
window.mainloop()

#import tkinter
# from utils import moment

#window=tkinter.Tk()
#window.title("GUI")
#window.geometry('600x600')
#lbl=tkinter.Label(window, text="Hello World!")
#lbl.place(x=0, y=0)
#bt = tkinter.Button(window,text="Enter",command=window.destroy)
#bt.place(x=0, y=50)
#bt1 = tkinter.Button(window,text="testme",command=window.destroy)
#bt1.place(x=0, y=100)

#window.mainloop()
