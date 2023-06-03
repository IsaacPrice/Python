from tkinter import *

# This makes the window with a name and specific window size
root = Tk()
root.title('My First Window')
root.geometry('600x400')

# This creates a label as a child of the root
text = Label(root, text='Hello, World!')

# This makes the window resize to the size of the text
text.pack()

# This will run until the user exits the window
root.mainloop()