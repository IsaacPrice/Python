from tkinter import *

# create the window
root = Tk()

# set window title and dimensions
root.title('this window has a button')
root.geometry('600x400')

# add a label
prompt_label = Label(root, text = 'Do you want the window to explode?')
prompt_label.grid()

# adding in an entry widget
text_entry = Entry(root, width = 10)
text_entry.grid(column = 1, row = 0)

# function to be called when button is clicked
def clicked():
    prompt_label.configure(text = 'I can\'t actually explode')

# create the button that calls the function on press
button_widget = Button(root, text = 'Click me', 
                       fg = 'red', command=clicked)

# set the button on the grid
button_widget.grid(column=1, row=0)

# run main loop
root.mainloop()