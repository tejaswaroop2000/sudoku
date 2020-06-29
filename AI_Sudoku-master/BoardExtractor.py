import cv2
import os
import numpy as np

class BoardExtractor:
    def __init__(self, imagepath):
        self.image = cv2.imread(imagepath, 0)
        self.originalimage = np.copy(self.image)
        self.extractedgrid = None

    def preprocess_image(self):

        gray = self.image

        gray = cv2.GaussianBlur(gray, (11, 11), 0)
        try:
            os.remove("StagesImages/1.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/1.jpg", gray)

        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C | cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 2)
        try:
            os.remove("StagesImages/2.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/2.jpg", gray)

        gray = cv2.bitwise_not(gray)
        try:
            os.remove("StagesImages/3.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/3.jpg", gray)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        gray = cv2.dilate(gray, kernel)
        self.image = gray
        try:
            os.remove("StagesImages/4.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/4.jpg", gray)
    def detect_and_crop_grid(self):

        outerbox = self.image
        maxi = -1
        maxpt = None
        value = 10
        try:
            os.remove("StagesImages/5.jpg")
        except:
            pass
        height, width = np.shape(outerbox)
        for y in range(height):
            row = self.image[y]
            for x in range(width):
                if row[x] >= 128:
                    area = cv2.floodFill(outerbox, None, (x, y), 64)[0]
                    if value > 0:
                        cv2.imwrite("StagesImages/5.jpg", outerbox)
                        value -= 1
                    if area > maxi:
                        maxpt = (x, y)
                        maxi = area

        cv2.floodFill(outerbox, None, maxpt, (255, 255, 255))
        for y in range(height):
            row = self.image[y]
            for x in range(width):
                if row[x] == 64 and x != maxpt[0] and y != maxpt[1]:
                    cv2.floodFill(outerbox, None, (x, y), 0)
        try:
            os.remove("StagesImages/6.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/6.jpg", outerbox)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        outerbox = cv2.erode(outerbox, kernel)
        try:
            os.remove("StagesImages/7.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/7.jpg", outerbox)
        lines = cv2.HoughLines(outerbox, 1, np.pi / 180, 200)
        def drawLine(line, img):
            height, width = np.shape(img)
            if line[0][1] != 0:
                m = -1 / np.tan(line[0][1])
                c = line[0][0] / np.sin(line[0][1])
                cv2.line(img, (0, int(c)), (width, int(m * width + c)), 255)
            else:
                cv2.line(img, (line[0][0], 0), (line[0][0], height), 255)
            return img
        tmpimg = np.copy(outerbox)
        for i in range(len(lines)):
            tmpimp = drawLine(lines[i], tmpimg)
        try:
            os.remove("StagesImages/8.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/8.jpg", tmpimg)
        def mergeLines(lines, img):
            height, width = np.shape(img)
            for current in lines:
                if current[0][0] is None and current[0][1] is None:
                    continue
                p1 = current[0][0]
                theta1 = current[0][1]
                pt1current = [None, None]
                pt2current = [None, None]
                if theta1 > np.pi * 45 / 180 and theta1 < np.pi * 135 / 180:
                    pt1current[0] = 0
                    pt1current[1] = p1 / np.sin(theta1)
                    pt2current[0] = width
                    pt2current[1] = -pt2current[0] / np.tan(theta1) + p1 / np.sin(theta1)
                else:
                    pt1current[1] = 0
                    pt1current[0] = p1 / np.cos(theta1)
                    pt2current[1] = height
                    pt2current[0] = -pt2current[1] * np.tan(theta1) + p1 / np.cos(theta1)
                for pos in lines:
                    if pos[0].all() == current[0].all():
                        continue
                    if abs(pos[0][0] - current[0][0]) < 20 and abs(pos[0][1] - current[0][1]) < np.pi * 10 / 180:
                        p = pos[0][0]
                        theta = pos[0][1]
                        pt1 = [None, None]
                        pt2 = [None, None]
                        if theta > np.pi * 45 / 180 and theta < np.pi * 135 / 180:
                            pt1[0] = 0
                            pt1[1] = p / np.sin(theta)
                            pt2[0] = width
                            pt2[1] = -pt2[0] / np.tan(theta) + p / np.sin(theta)
                        else:
                            pt1[1] = 0
                            pt1[0] = p / np.cos(theta)
                            pt2[1] = height
                            pt2[0] = -pt2[1] * np.tan(theta) + p / np.cos(theta)
                        if (pt1[0] - pt1current[0])**2 + (pt1[1] - pt1current[1])**2 < 64**2 and (pt2[0] - pt2current[0])**2 + (pt2[1] - pt2current[1])**2 < 64**2:
                            current[0][0] = (current[0][0] + pos[0][0]) / 2
                            current[0][1] = (current[0][1] + pos[0][1]) / 2
                            pos[0][0] = None
                            pos[0][1] = None
            lines = list(filter(lambda a : a[0][0] is not None and a[0][1] is not None, lines))
            return lines

        lines = mergeLines(lines, outerbox)
        topedge = [[1000, 1000]]
        bottomedge = [[-1000, -1000]]
        leftedge = [[1000, 1000]]
        leftxintercept = 100000
        rightedge = [[-1000, -1000]]
        rightxintercept = 0
        for i in range(len(lines)):
            current = lines[i][0]
            p = current[0]
            theta = current[1]
            xIntercept = p / np.cos(theta)
            if theta > np.pi * 80 / 180 and theta < np.pi * 100 / 180:
                if p < topedge[0][0]:
                    topedge[0] = current[:]
                if p > bottomedge[0][0]:
                    bottomedge[0] = current[:]
            if theta < np.pi * 10 / 180 or theta > np.pi * 170 / 180:
                if xIntercept > rightxintercept:
                    rightedge[0] = current[:]
                    rightxintercept = xIntercept
                elif xIntercept <= leftxintercept:
                    leftedge[0] = current[:]
                    leftxintercept = xIntercept

        tmpimg= np.copy(outerbox)
        tmppp = np.copy(self.originalimage)
        tmppp = drawLine(leftedge, tmppp)
        tmppp = drawLine(rightedge, tmppp)
        tmppp = drawLine(topedge, tmppp)
        tmppp = drawLine(bottomedge, tmppp)

        tmpimg = drawLine(leftedge, tmpimg)
        tmpimg = drawLine(rightedge, tmpimg)
        tmpimg = drawLine(topedge, tmpimg)
        tmpimg = drawLine(bottomedge, tmpimg)
        try:
            os.remove("StagesImages/9.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/9.jpg", tmpimg)

        leftedge = leftedge[0]
        rightedge = rightedge[0]
        bottomedge = bottomedge[0]
        topedge = topedge[0]

        left1 = [None, None]
        left2 = [None, None]
        right1 = [None, None]
        right2 = [None, None]
        top1 = [None, None]
        top2 = [None, None]
        bottom1 = [None, None]
        bottom2 = [None, None]

        if leftedge[1] != 0:
            left1[0] = 0
            left1[1] = leftedge[0] / np.sin(leftedge[1])
            left2[0] = width
            left2[1] = -left2[0] / np.tan(leftedge[1]) + left1[1]
        else:
            left1[1] = 0
            left1[0] = leftedge[0] / np.cos(leftedge[1])
            left2[1] = height
            left2[0] = left1[0] - height * np.tan(leftedge[1])

        if rightedge[1] != 0:
            right1[0] = 0
            right1[1] = rightedge[0] / np.sin(rightedge[1])
            right2[0] = width
            right2[1] = -right2[0] / np.tan(rightedge[1]) + right1[1]
        else:
            right1[1] = 0
            right1[0] = rightedge[0] / np.cos(rightedge[1])
            right2[1] = height
            right2[0] = right1[0] - height * np.tan(rightedge[1])

        bottom1[0] = 0
        bottom1[1] = bottomedge[0] / np.sin(bottomedge[1])

        bottom2[0] = width
        bottom2[1] = -bottom2[0] / np.tan(bottomedge[1]) + bottom1[1]

        top1[0] = 0
        top1[1] = topedge[0] / np.sin(topedge[1])
        top2[0] = width
        top2[1] = -top2[0] / np.tan(topedge[1]) + top1[1]
        
        leftA = left2[1] - left1[1]
        leftB = left1[0] - left2[0]
        leftC = leftA * left1[0] + leftB * left1[1]

        rightA = right2[1] - right1[1]
        rightB = right1[0] - right2[0]
        rightC = rightA * right1[0] + rightB * right1[1]

        topA = top2[1] - top1[1]
        topB = top1[0] - top2[0]
        topC = topA * top1[0] + topB * top1[1]

        bottomA = bottom2[1] - bottom1[1]
        bottomB = bottom1[0] - bottom2[0]
        bottomC = bottomA * bottom1[0] + bottomB * bottom1[1]

        detTopLeft = leftA * topB - leftB * topA

        ptTopLeft = ((topB * leftC - leftB * topC) / detTopLeft, (leftA * topC - topA * leftC) / detTopLeft)
        
        detTopRight = rightA * topB - rightB * topA

        ptTopRight = ((topB * rightC - rightB * topC) / detTopRight, (rightA * topC - topA * rightC) / detTopRight)

        detBottomRight = rightA * bottomB - rightB * bottomA

        ptBottomRight = ((bottomB * rightC - rightB * bottomC) / detBottomRight, (rightA * bottomC - bottomA * rightC) / detBottomRight)
        
        detBottomLeft = leftA * bottomB - leftB * bottomA

        ptBottomLeft = ((bottomB * leftC - leftB * bottomC) / detBottomLeft,
                               (leftA * bottomC - bottomA * leftC) / detBottomLeft)
        cv2.circle(tmppp, (int(ptTopLeft[0]), int(ptTopLeft[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptTopRight[0]), int(ptTopRight[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptBottomLeft[0]), int(ptBottomLeft[1])), 5, 0, -1)
        cv2.circle(tmppp, (int(ptBottomRight[0]), int(ptBottomRight[1])), 5, 0, -1)
        try:
            os.remove("StagesImages/10.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/10.jpg", tmppp)

        leftedgelensq = (ptBottomLeft[0] - ptTopLeft[0])**2 + (ptBottomLeft[1] - ptTopLeft[1])**2
        rightedgelensq = (ptBottomRight[0] - ptTopRight[0])**2 + (ptBottomRight[1] - ptTopRight[1])**2
        topedgelensq = (ptTopRight[0] - ptTopLeft[0])**2 + (ptTopLeft[1] - ptTopRight[1])**2
        bottomedgelensq = (ptBottomRight[0] - ptBottomLeft[0])**2 + (ptBottomLeft[1] - ptBottomRight[1])**2
        maxlength = int(max(leftedgelensq, rightedgelensq, bottomedgelensq, topedgelensq)**0.5)
        src = [(0, 0)] * 4
        dst = [(0, 0)] * 4
        src[0] = ptTopLeft[:]
        dst[0] = (0, 0)
        src[1] = ptTopRight[:]
        dst[1] = (maxlength - 1, 0)
        src[2] = ptBottomRight[:]
        dst[2] = (maxlength - 1, maxlength - 1)
        src[3] = ptBottomLeft[:]
        dst[3] = (0, maxlength - 1)
        src = np.array(src).astype(np.float32)
        dst = np.array(dst).astype(np.float32)
        self.extractedgrid = cv2.warpPerspective(self.originalimage, cv2.getPerspectiveTransform(src, dst), (maxlength, maxlength))
        try:
            os.remove("StagesImages/11.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/11.jpg", self.extractedgrid)
        self.extractedgrid = cv2.resize(self.extractedgrid, (252, 252))

    def create_image_grid(self):
        if self.extractedgrid is None:
            raise Exception("Grid not yet extracted")
        grid = np.copy(self.extractedgrid)
        edge = np.shape(grid)[0]
        celledge = edge // 9

        grid = cv2.bitwise_not(cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 101, 1))
        try:
            os.remove("StagesImages/12.jpg")
        except:
            pass
        cv2.imwrite("StagesImages/12.jpg", grid)
        tempgrid = []
        for i in range(celledge, edge+1, celledge):
            for j in range(celledge, edge+1, celledge):
                rows = grid[i-celledge:i]
                tempgrid.append([rows[k][j-celledge:j] for k in range(len(rows))])

        finalgrid = []
        for i in range(0, len(tempgrid)-8, 9):
            finalgrid.append(tempgrid[i:i+9])

        #Converting all the cell images to np.array
        for i in range(9):
            for j in range(9):
                finalgrid[i][j] = np.array(finalgrid[i][j])
        try:
            for i in range(9):
                for j in range(9):
                    os.remove("BoardCells/cell"+str(i)+str(j)+".jpg")
        except:
            pass
        for i in range(9):
            for j in range(9):
                cv2.imwrite(str("BoardCells/cell"+str(i)+str(j)+".jpg"), finalgrid[i][j])
        return finalgrid


