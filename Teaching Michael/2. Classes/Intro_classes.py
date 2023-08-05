# This is a program to introduce classes in Python
class Animal:
    def eat(self, food):
        print("I can eat", food)

    def sleep(self):
        print("I can sleep")

    def make_sound(self, sound="woof"):
        print(sound)


# Create an Animal object
dog = Animal()

# Call the sleep method
dog.sleep()

# Call the eat method
dog.eat("dog food")

# Call the make_sound method
dog.make_sound("woof woof")



    