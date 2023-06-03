from tkinter import *

# create root window
root = Tk()

# set the properties of the window
root.title('Making Menu')
root.geometry('600x400')

# creating a menu bar
# adding a new item to the menu bar
# adding in items into the item inside of the menu bar
menu = Menu(root)
item = Menu(menu)
item.add_command(label='New')
menu.add_cascade(label='File', menu=item)
root.config(menu=menu)

# adding a label to the window
text_label = Label(root, width = 10)
text_label.grid()

# adding an entry field
text_input = Entry(root, width=10)
text_input.grid(column = 1, row = 0)

# a function to be called when the user presses the button
def clicked():

    # gets the users input and changes the label
    combined = 'You wrote ' + text_input.get()
    text_label.configure(text = combined)

# create button widget and put it on the grid
enter_button = Button(root, text = 'Enter',
                      bg = 'blue', command = clicked)
enter_button.grid(column = 2, row = 0)

# run the main loop
root.mainloop()