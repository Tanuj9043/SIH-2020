import socket
import time
from socketIO_client import SocketIO, LoggingNamespace

socket =  SocketIO('localhost', 8080, LoggingNamespace)
print('Connection started')

i = 0
while True:
    time.sleep(2)
    socket.emit('keyApp',i)
    socket.emit('keyApp2',i**2)
    print(i)
    i += 1
