import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QMainWindow, QAction, QMenu, QMenuBar
from PyQt5.QtCore import Qt

class MyWindow(QWidget):
    def __init__(self):
        super(MyWindow, self).__init__()

        # Create widgets
        self.label = QLabel('Enter something:')
        self.textbox = QLineEdit(self)
        self.button = QPushButton('Click Me!', self)
        
        # Connect button to function
        self.button.clicked.connect(self.on_click)
        
        # Create layout and add widgets
        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addWidget(self.textbox)
        vbox.addWidget(self.button)
        
        # Set dialog layout
        self.setLayout(vbox)

    def on_click(self):
        textbox_value = self.textbox.text()
        self.label.setText(f'You entered: {textbox_value}')
        self.textbox.setText("")


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Well-designed PyQt Window')
        self.setGeometry(100, 100, 500, 300)

        # Create a menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')
        help_menu = menubar.addMenu('Help')

        # Add actions to menus
        exit_action = QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        # Set central widget
        self.central_widget = MyWindow()
        self.setCentralWidget(self.central_widget)
    
    def show_about(self):
        self.central_widget.label.setText("This is a well-designed PyQt app.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
