import tkinter as tk
from PIL import ImageTk, Image

class Frames:
    def __init__(self, parent, width, height, x=10, y=10):
        self.winFrame = tk.Frame(parent, width=width, height=height, borderwidth=5, relief="ridge")
        self.winFrame.place(x=x, y=y)

    def addWidget(self, widget, x, y):
        widget.place(x=x, y=y)

    def hide(self):
        self.winFrame.place_forget()

    def show(self):
        self.winFrame.place(x=10, y=10)
