from binance.client import Client

client = Client("NPXlmdB5YkQmjsmv2ZrtrCirHVQSBIhQMOkBfA4NkfBQVkzA8XK2Hp9N53E5WPoo", "Iwlz5wVmszSThrvzGXdiIWcdQ6AvF7hA7cpoY7LxGLKMIHKObKqTGa8wmkbmXbUU")
print(client.get_account_status())  
