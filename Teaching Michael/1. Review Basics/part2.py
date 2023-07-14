# Gets the users name
name = input("Enter your name: ")
age = input("Enter your age: ")

# Print the variables
print(name, age)

# Prints the types of the variables
print(type(name), type(age))

# If we want to use the age in some sort of calculation, we need to get it to an int, not a string
age = int(age)
print(age, type(age))

# More on the input, It will always give a string, and only has one parameter, the prompt
num1 = int(input("Enter a number: "))
num2 = int(input("Enter another number: "))

# All of the basic operators
sum = num1 + num2
difference = num1 - num2
product = num1 * num2
quotient = num1 / num2

# Other operators
remainder = num1 % num2
floor_division = num1 // num2 # Gives how many whole numbers fit into the number
exponent = num1 ** num2

# Print the result
print(floor_division)