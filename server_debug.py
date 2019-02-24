import socket
ip_port = ('127.0.0.1',8080)
sk = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sk.bind(ip_port)
sk.listen(5)

while True:
    clientsocket, addr = sk.accept()
    data = clientsocket.recv(1024000)
    print(data.decode())
    clientsocket.close()