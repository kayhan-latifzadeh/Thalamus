from abc import ABC, abstractmethod

class RecordingDevice(ABC):
    def __init__(self, device_id, server_host='localhost', server_port=9000):
        self.device_id = device_id
        self.server_host = server_host
        self.server_port = server_port

    @abstractmethod
    def start(self):
        pass