import socket
import threading

class RendezvousServer:
    def __init__(self, host='192.168.2.8', port=1234):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connections = []
        
    def start(self):
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)  # Listen for up to 5 incoming connections
        print(f"Rendezvous server is listening on {self.host}:{self.port}")
        
        while True:
            client_socket, address = self.server_socket.accept()
            print(f"New connection from {address}")
            self.connections.append(client_socket)
            
            thread = threading.Thread(target=self.handle_client, args=(client_socket,))
            thread.start()
    
    def handle_client(self, client_socket):
        while True:
            try:
                data = client_socket.recv(1024)
                if not data:
                    break
                    
                # Here you can handle the received data
                print(f"Received data: {data}")
                
                # Send back a response or broadcast to other clients
                response = b"Received your message!"
                client_socket.sendall(response)
                    
            except Exception as e:
                print(f"Error handling client: {e}")
                break
                    
        client_socket.close()
        self.connections.remove(client_socket)

    def stop(self):
        for connection in self.connections:
            connection.close()
        self.server_socket.close()

if __name__ == "__main__":
    server = RendezvousServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("Stopping server...")
        server.stop()
