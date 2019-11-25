import os
import math
import cv2

class VideoToFrames:
    def __init__(self):
        print("Video To Frames Ready to run!")
    def run(self):
        listing = os.listdir("video/")
        fgbg = cv2.createBackgroundSubtractorMOG2(history=50 , detectShadows = False)
        count = 1
        for file in listing:
            video = cv2.VideoCapture("video/" + file)
            print(video.isOpened())
            framerate = video.get(cv2.CAP_PROP_FPS)
            print(framerate)
            os.makedirs("video/" + "video_" + str(int(count)))
            i=0
            while (video.isOpened()):
                frameId = video.get(1)
                success,image = video.read()
                success, image = video.read()
                if( image is not None ):
                    image=cv2.resize(image,(224,224), interpolation = cv2.INTER_AREA)
                    image = fgbg.apply(image, learningRate=0.5)
                if (success != True):
                    break
                filename = "video/video_" + str(int(count)) + "/image_" + str(i + 1) + ".jpg"
                i=i+1
                print(filename)
                cv2.imwrite(filename,image)
            video.release()
            print('done')
            count+=1