# This program will create a book class

class Book:
    # The __init__ method is a special method that is called when you create an object, and self is a reference to the object
    def __init__(self, title, author, pages, price):
        self.title = title
        self.author = author
        self.pages = pages
        self.price = price

    # The __str__ method is a special method that is called when you print an object
    def __str__(self):
        return f"{self.title} by {self.author}"
    
    def get_price(self):
        return self.price


# Create some book objects
b1 = Book("War and Peace", "Leo Tolstoy", 1225, 39.95)
b2 = Book("The Catcher in the Rye", "JD Salinger", 234, 29.95)

# Print the book objects
print(b1)
print(b2)
