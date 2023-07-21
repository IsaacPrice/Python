def add_ten(something):
    return something + 10

num = 3
num = add_ten(num)

def print_greeting(name="Joe Rogan"):
    print("Hello! My name is", name + ".")


#print_greeting("Isaac Price")
#print_greeting()

# I dont really know, but I think the "-> str" doesn't do anything other than to visualize what is supposed to be returned
def something(boolen) -> str:
    if (boolen):
        return "string"
    else:
        return "0"
    
print(type(something(False)))

