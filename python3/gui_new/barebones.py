# The code for changing pages was derived from: http://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter
# License: http://creativecommons.org/licenses/by-sa/3.0/

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import ImageTk, Image

import tkinter as tk
from tkinter import ttk

from glob import glob

LARGE_FONT= ("Verdana", 12)


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        # tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Sea of BTC client")


        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        print(PageOne)
        for F in (StartPage, PageOne):

            #if F == PageOne:
                #self.frames[F] = F(container, self)
                #for i,_ in enumerate(glob("../../plots/*.png")):
                    #frame = F(container, self, i)
                    #self.frames[i] = frame
            #else:
                #frame = F(container, self)
                #self.frames[F] = frame


            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):
        #print(cont)
        #print(self.frames)
        #print(self.frames[cont])

        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Start Page", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Visit Page 1",
                            command=lambda: controller.show_frame(PageOne))
                            #command=lambda: controller.show_frame(0))
        button.pack()

        button2 = ttk.Button(self, text="Visit Page 2",
                            command=lambda: controller.show_frame(PageOne))
        button2.pack()

        #button3 = ttk.Button(self, text="Graph Page",
                            #command=lambda: controller.show_frame(PageThree))
        #button3.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller, index=0):
    #def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)

        self.plots=[ImageTk.PhotoImage(Image.open(fileName)) for fileName in glob("../../plots/*.png")]
        if index < 1:
            self.index=0
        elif index < len(self.plots):
            self.index=index
        else:
            self.index=len(self.plots)

        #print(self.index)
        #label = tk.Label(self, text="Page One!!!", font=LARGE_FONT)
        #label.pack(pady=10,padx=10)

        #print(self.plots[self.index])
        #plot=ImageTk.PhotoImage(Image.open("../../plots/kmh.png"))
        plot=self.plots[self.index]
        #plot=ImageTk.PhotoImage(Image.open(self.plots[0]))
        #plot=ImageTk.PhotoImage(Image.open(self.plots[self.index]))
        self.image = plot

        label = tk.Label(self, text="test", image=plot, width=1000, height=800)
        self.label = label
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()


        #button2 = ttk.Button(self, text="Next plot",
                            #command=lambda: controller.show_frame(PageTwo))
        #button2.pack()

        #button3 = ttk.Button(self, text="Previous plot",
                            #command=lambda: controller.show_frame(PageTwo))
        #button3.pack()

        nextButton = ttk.Button(self, text="next",
                            command=lambda *args: self.show_next(parent, controller))
        self.nextButton = nextButton
        nextButton.pack()

        prevButton = ttk.Button(self, text="previous",
                            command=lambda *args: self.show_previous(parent, controller))
        self.prevButton = prevButton
        prevButton.pack()

    def show_next(self, parent, controller):
        self.incrementIndex()
        self.label.configure(image=self.plots[self.index])
        #newPage=PageOne(parent, controller, self.index)
        #controller.show_frame(self.index)
        if self.index==len(self.plots)-1:
            self.nextButton.forget()
        if self.index==1:
            self.prevButton.pack()
        controller.show_frame(PageOne)

    def show_previous(self, parent, controller):
        self.decrementIndex()
        self.label.configure(image=self.plots[self.index])
        #newPage=PageOne(parent, controller, self.index)
        #controller.show_frame(self.index)
        if self.index==0:
            self.prevButton.forget()
        if self.index==len(self.plots)-1:
            self.nextButton.pack()
        controller.show_frame(PageOne)

    def incrementIndex(self):
        newIndex=self.index+1
        if newIndex < len(self.plots):
            self.index=newIndex
        else:
            self.index=len(self.plots)-1

    def decrementIndex(self):
        newIndex=self.index-1
        if newIndex >= 0:
            self.index=newIndex
        else:
            self.index=0




app = SeaofBTCapp()
app.mainloop()
