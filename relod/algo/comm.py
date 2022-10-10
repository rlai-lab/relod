import pickle

class MODE:
    LOCAL_ONLY = 'local only'
    REMOTE_ONLY = 'remote only'
    REMOTE_LOCAL = 'remote local'
    EVALUATION = 'evaluation'

def recv_message(client_sock):
    bytes_to_recv = 4 # recieve message length first
    message_buffer = bytearray()
    while bytes_to_recv > 0:
        chunk = client_sock.recv(bytes_to_recv)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        message_buffer += chunk
        bytes_to_recv -= len(chunk)
    
    bytes_to_recv = int.from_bytes(message_buffer, byteorder='big')
    message_buffer = bytearray() # recieve actual parameters
    while bytes_to_recv > 0:
        chunk = client_sock.recv(bytes_to_recv)
        if chunk == b'':
            raise RuntimeError("socket connection broken")
        message_buffer += chunk
        bytes_to_recv -= len(chunk)
            
    return pickle.loads(message_buffer)

def send_message(mess, client_sock):
    mess = pickle.dumps(mess)
    length = (len(mess)).to_bytes(4, 'big')
    client_sock.sendall(length)
    client_sock.sendall(mess)
