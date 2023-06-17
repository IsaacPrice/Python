from pynput import keyboard
import pandas as pd
import pickle
import time

unedited = pd.read_csv("food.csv")

print("Press y to keep, anything else to discard")

edited = unedited
x = 0

while x < len(unedited):

    print(unedited.loc[x, 'Description'], end=": ")

    with keyboard.Events() as events:
        # Block for as much as possible
        event = events.get(1e6)
        if event.key == keyboard.KeyCode.from_char('y'):
            print("kept")
        elif event.key == keyboard.KeyCode.from_char('u'):
            x -= 2
            print("went back")
        else:
            print("didn't keep")
            edited.drop(x, axis="index")
    x += 1
    time.sleep(.2)