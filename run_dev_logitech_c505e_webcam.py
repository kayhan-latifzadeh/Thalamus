import json
import socket
import time
import random
from device_interface import RecordingDevice
import pandas as pd
import cv2
import base64
from io import BytesIO
from PIL import Image

def get_frame_as_base64(file_path, frame_num):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {file_path}")

        target_frame = int(frame_num)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        cap.release()

        if not success:
            raise ValueError(f"Could not read frame number {target_frame}")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)

        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return base64_str

class SimulatedDevice(RecordingDevice):
    def __init__(self, device_id, interval=1.0, **kwargs):
        super().__init__(device_id, **kwargs)
        self.interval = interval
        self.video_path = 'pre_recorded_files/webcam_covered.mp4'
        self.frame_number = 0
        self.origin_ts = 1690535469479

    

    def get_data_entry(self):

        data_entry = {'image': get_frame_as_base64(self.video_path, self.frame_number),
                      'timestamp': (self.origin_ts + int(self.frame_number * 33.33)) }
        self.frame_number += 1
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
    sampling_frequency = 30 # Hz
    interval = (1/sampling_frequency) # in seconds
    device = SimulatedDevice(device_id='logitech_c505e_webcam', interval=interval)
    device.start()
