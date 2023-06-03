from tkinter import *

# create the window
root = Tk()

# define properties of the window
root.title('Text entry')
root.geometry('600x400')

# adding in the label
text_label = Label(root, text = 'Enter your exact location.')
text_label.grid()

# adding a Entry thingy
text_entry = Entry(root, width=10) 
text_entry.grid(column = 1, row = 0)

# function to call every time the button is clicked
def clicked():

    # gets what the user wrote and sets it to the label
    combined = 'You wrote ' + text_entry.get()
    text_label.configure(text = combined)

# create a button widget and set the grid
enter_button = Button(root, text = 'Enter your response', 
                      fg = 'blue', command = clicked)
enter_button.grid(column = 2, row = 0)

# Run the main loop
root.mainloop()
