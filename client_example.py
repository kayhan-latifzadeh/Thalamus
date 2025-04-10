import socket
import json

def start_client(subscribe_to=['device_01'], server_host='localhost', server_port=9001):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((server_host, server_port))
        sub_msg = json.dumps({'subscribe': subscribe_to})
        sock.sendall(sub_msg.encode('utf-8'))
        print("Subscribed to:", subscribe_to)
        try:
            while True:
                data = sock.recv(1024)
                if not data:
                    break
                print("Received:", data.decode('utf-8').strip())
        except KeyboardInterrupt:
            print("\nClient exiting.")

if __name__ == '__main__':
    start_client(['unicorn_hybrid_black_eeg', 'gp3_eye_tracker', 'logitech_c505e_webcam'])
