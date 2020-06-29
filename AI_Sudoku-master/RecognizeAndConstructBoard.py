import cv2
import numpy as np
from Recognizer import DigitRecognizer
import os

class ConstructGrid:

    def __init__(self, cellarray, modeltype):
        self.cellarray = cellarray
        self.recognizer = DigitRecognizer(modeltype)
        self.finalgrid = [[0 for i in range(9)] for j in range(9)]
        self.imagewritten = False
    def constructgrid(self):
        threshold = 5*255
        for i in range(9):
            for j in range(9):
                #First preprocess and clean the board cell
                tmp = np.copy(self.cellarray[i][j])
                tmp = self.recognizer.preprocess_image(tmp)
                cv2.imwrite("CleanedBoardCells/cell"+str(i)+str(j)+".jpg", tmp)
                #Checking if the board cell is empty or not
                finsum = 0
                for k in range(28):
                    rowsum = sum(tmp[k])
                    finsum += rowsum
                if finsum < threshold:
                    self.finalgrid[i][j] = 0
                    continue
                if not self.imagewritten:
                    try:
                        os.remove("StagesImages/13.jpg")
                        os.remove("StagesImages/14.jpg")
                    except:
                        pass
                    cv2.imwrite("StagesImages/13.jpg", self.cellarray[i][j])
                    cv2.imwrite("StagesImages/14.jpg", tmp)
                    self.imagewritten = True
                pred = self.recognizer.make_prediction(str("CleanedBoardCells/cell"+str(i)+str(j)+".jpg"))
                self.finalgrid[i][j] = int(pred)
        return self.finalgrid




