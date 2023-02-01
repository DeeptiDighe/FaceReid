from threading import Thread

from Producer import Producer

def produce(camera):
    producer = Producer(camera['id'], camera['url'])
    producer.start()


if __name__ == '__main__':
    camera = dict()
    camera[1] = {'id':1, 'url':r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\videos\v1.mp4'}
    #camera[2] = {'id':2, 'url':r"C:\Users\Accubits\Pictures\v1.mp4"}
    #camera[2] = {'id':2, 'url':r'C:\Projects\Emotyx\TrainedModels\Fully-Automated-red-light-Violation-Detection-master\videos\aziz1.mp4'}
    #camera['id'] = 1
    #camera['url'] = r'C:\Projects\Emotyx\Code\FaceReidentification\Complete\videos\v1.mp4'
    #camera['id'] = 2
    #camera['url'] = r'C:\Projects\Emotyx\TrainedModels\Fully-Automated-red-light-Violation-Detection-master\videos\aziz1.mp4'
    for cam in camera:
        thread = Thread(target=produce, args=[camera[cam]])
        thread.start()