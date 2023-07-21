# Python lists are ordered, changeable, and allow duplicate values
fruits = ["apple", "banana", "orange"]

# Access the values with [], indexes start at 0
print(fruits[1])
print(fruits)

# To change the values, access them with [] and assign a value
fruits[2] = "grape"
print(fruits)

# Lists can hold any values, in the same list
my_list = ["grape", 10, False, "apple", .78]
print(my_list)


# Tuples are ordered, unchangeable, and allow duplicate values
my_tuple = ("apple", "banana", "orange")
print(my_tuple)

# Accessing the values is the same as lists
print(my_tuple[0])

# Can get the length of a tuple, set, or list with "len(var)"
print(len(my_list), len(my_tuple))


# Dictonarys are ordered, changeable, and do not allow duplicates
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)