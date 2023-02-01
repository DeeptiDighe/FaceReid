from PIL import Image
from ThreadedCamera import ThreadedCamera
import app
import cv2
import base64

def convert_frame_to_bin(frame):
    img = Image.fromarray(frame)
    bytes = img.tobytes()
    return base64.b64encode(bytes).decode(), img.size

def rescale_frame(frame, percent=20):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def convert_frame_to_bino(frame):
    retval, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode("utf-8")

class Producer:
    def __init__(self, id, url):
        self.id = id
        self.url = url

    def start(self):
        streamer = ThreadedCamera(self.url)
        i=0
        while True:
            try:
                i+=1
                frame = streamer.grab_frame()
                if frame is None or i%10!=0:
                    continue

                frame_binary = convert_frame_to_bino(frame)
                pid = app.call_verify_method(frame_binary, self.id, (frame.shape[1], frame.shape[0]))
                status = app.get_status(pid)
                result = app.task_result(pid)
                print(result)
                

            except Exception as e:
                print("ERROR WHILE FETCHING FRAME ", e)
                cap = cv2.VideoCapture(self.url)
