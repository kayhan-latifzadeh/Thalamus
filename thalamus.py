import socket
import threading
import json
from collections import defaultdict
import time
import argparse

parser = argparse.ArgumentParser(description="Run Thalamus server with specified ports")
parser.add_argument('--device-port', type=int, default=9000, help='Port for device connection')
parser.add_argument('--client-port', type=int, default=9001, help='Port for client connection')


args = parser.parse_args()

clients = defaultdict(list)  # device_id -> list of client sockets

def handle_device(conn, addr):
    with conn:
        buffer = b''
        while True:
            data = conn.recv(1024)
            if not data:
                break
            buffer += data
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                try:
                    msg = json.loads(line.decode('utf-8'))
                    device_id = msg.get('device_id')
                    if device_id in clients:
                        for client_sock in clients[device_id]:
                            try:
                                client_sock.sendall(line + b'\n')
                            except Exception:
                                pass  # Handle broken pipe if client disconnected
                except Exception as e:
                    print("Error parsing message:", e)

def handle_client(conn, addr):
    with conn:
        try:
            data = conn.recv(1024).decode('utf-8')
            sub = json.loads(data)
            device_ids = sub.get('subscribe', [])
            for device_id in device_ids:
                clients[device_id].append(conn)
            print(f"Client at {addr} subscribed to: {device_ids}")

            # Just keep the connection open; no need to recv again
            while True:
                time.sleep(1)
        except Exception as e:
            print("Client handler error:", e)
        finally:
            for lst in clients.values():
                if conn in lst:
                    lst.remove(conn)

def start_server(device_port=args.device_port, client_port=args.client_port):
    def acceptor(label, port, handler):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('0.0.0.0', port))
            s.listen()
            print(f"{label}: listening on port {port}")
            while True:
                conn, addr = s.accept()
                print(f"Accepted connection from {addr} on port {port}")
                threading.Thread(target=handler, args=(conn, addr), daemon=True).start()

    threading.Thread(target=acceptor, args=('Devices', device_port, handle_device), daemon=True).start()
    threading.Thread(target=acceptor, args=('Clients', client_port, handle_client), daemon=True).start()

    print("Thalamus server started. Press Ctrl+C to stop.")
    while True:
        try:
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down server.")
            break

if __name__ == '__main__':
    start_server()
