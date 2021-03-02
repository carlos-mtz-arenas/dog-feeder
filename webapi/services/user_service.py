import os

default_user = os.environ['API_USER']
default_pwd = os.environ['API_PWD']

def authenticate(user, password):
    # TODO: write a proper login with a DB or something
    #  might be useful for the client to have some information
    if (user != default_user or password != default_pwd):
        raise Exception('Invalid credentials')
    return True