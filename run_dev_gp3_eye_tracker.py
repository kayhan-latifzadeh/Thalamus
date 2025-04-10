import json
import socket
import time
from device_interface import RecordingDevice
import pandas as pd

class SimulatedDevice(RecordingDevice):
    def __init__(self, device_id, interval=1.0, **kwargs):
        super().__init__(device_id, **kwargs)
        self.interval = interval
        self.df = pd.read_csv('pre_recorded_files/eye-tracking.csv')
        self.entry_index = 0

    def get_data_entry(self):
        data_entry = self.df.iloc[self.entry_index].to_dict()
        self.entry_index += 1
        return data_entry

    def start(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.server_host, self.server_port))
            print(f"[{self.device_id}] Connected to server at {self.server_host}:{self.server_port}")
            while True:
                entry = self.get_data_entry()
                entry['device_id'] = self.device_id
                msg = json.dumps(entry)
                print(f"[{self.device_id}] Sending: {msg}")
                try:
                    sock.sendall(msg.encode('utf-8') + b'\n')
                except BrokenPipeError:
                    print("Connection to server lost.")
                    break
                time.sleep(self.interval)

if __name__ == '__main__':
    sampling_frequency = 150 # Hz
    interval = (1/sampling_frequency) # in seconds
    device = SimulatedDevice(device_id='gp3_eye_tracker', interval=interval)
    device.start()
