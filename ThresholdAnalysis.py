import cv2 as cv
import copy, csv
import numpy as np


file = open("star.csv", "w", newline="")
writer = csv.writer(file)

class contourFinder:
    def __init__(self, filename) -> None:
        self.__image = cv.imread(str(filename))
        self.__preProcessing()

    def __preProcessing(self):
        self.__gray = cv.cvtColor(self.__image, cv.COLOR_BGR2GRAY)
        self.__blurred = cv.GaussianBlur(self.__gray, (5, 5), 0)

    def __sharpenImage(self):
        __kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
        self.__sharpened = cv.filter2D(self.__image, -1, __kernel)

        return self.__sharpened

    def findContours(self, t1, t2):
        tmp = self.__sharpenImage()
        self.__canneyed = cv.Canny(self.__blurred, t1, t2)
        (self.cnts, _) = cv.findContours(self.__canneyed, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        self.clone = copy.deepcopy(self.__image)

        for c in self.cnts:
            M = cv.moments(c)

            cX, cY = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

            cv.circle(self.clone, (cX, cY), 5, (1, 227, 254), -1)

if __name__ == "__main__":
    image = contourFinder(input("Input file name: "))

    for t2 in range(100, 900, 10):
        for r in np.arange(0.33, 0.50, 0.05):
            try:
                image.findContours((t2*r)//1, t2)
                writer.writerow([(t2*r)//1, t2, r, len(image.cnts)])
                print(f"{(t2*r//1)}, {t2}, {len(image.cnts)}")
            except ZeroDivisionError:
                continue
    file.close()
