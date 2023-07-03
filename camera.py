# Import essential libraries
import requests
import cv2
import numpy as np
import imutils


class camera:
    def __init__(self, url=""):
        self.url = url
        self.distances = []

    def set_url(self, url):
        self.url = url

    def start_streaming(self):
        while True:
            img_resp = requests.get(self.url)
            img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_arr, -1)

            # resize image
            img2 = imutils.resize(img, width=1000, height=1800)

            # process image
            img3 = self.process_image(img2)

            cv2.imshow("Android_cam", img3)

            # Press Esc key to exit
            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

    def process_image(self, img):
        image = img.copy()
        # get to grayscale
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # process

        img2 = cv2.GaussianBlur(imgray, (11, 11), 2)
        img2 = cv2.bilateralFilter(img2, 15, 50, 50)
        img2 = cv2.threshold(img2, 100, 256, cv2.THRESH_BINARY)[1]

        # get contours
        ret, thresh = cv2.threshold(img2, 127, 255, 0)
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # get and draw bounding box

        # first draw a circle in the middle of the image
        middle = (round(image.shape[1] / 2), round(image.shape[0] / 2))
        cv2.circle(
            image,
            middle,
            radius=15,
            color=(255, 0, 0),
            thickness=-1,
        )

        self.distances = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1300 and area < 100000:
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                # print(len(approx))
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                rect_center = (round((x + w / 2)), round((y + h / 2)))
                cv2.circle(
                    image,
                    rect_center,
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                )
                cv2.line(image, middle, rect_center, (0, 0, 255), 2)

                self.distances.append(
                    (rect_center[0] - middle[0], rect_center[1] - middle[1])
                )

        print(self.distances)

        # draw contours
        # cv2.drawContours(
        #     image, contours, -1, (0, 255, 0), thickness=2, lineType=cv2.LINE_AA
        # )

        return image


camera = camera("http://192.168.43.1:8080/shot.jpg")

camera.start_streaming()
