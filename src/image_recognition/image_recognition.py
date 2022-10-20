import cv2
import imutils
import numpy
# from tensorflow import keras
# from keras.models import load_model

def get_sudoku_board_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edges = cv2.Canny(bfilter, 30, 180)
    kernel = numpy.ones((3,3),numpy.uint8)
    edges = cv2.dilate(edges,kernel,iterations = 1)
    # kernel = numpy.ones((5,5),numpy.uint8)
    # edges = cv2.erode(edges,kernel,iterations = 1)
    keypoints = cv2.findContours(edges.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg)
    return contours

def get_perspective(img, location, height = 900, width = 900):
    pts1 = numpy.float32([location[0], location[3], location[1], location[2]])
    pts2 = numpy.float32([[0, 0], [width, 0], [0, height], [width, height]])
    # Apply Perspective Transform Algorithm
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(img, matrix, (width, height))
    return result

def adjust_for_perspective(img, contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    # Finds rectangular contour
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    # if location == None:
    #     raise Exception("board not found")
    result = get_perspective(img, location)
    return result, location


if __name__ == "__main__":
    img = cv2.imread('src/image_recognition/test')

    # img = cv2.imread('src/image_recognition/easypuzzle.png')
    # img = cv2.imread('src/image_recognition/sudoku-angle.png')
    # img = cv2.imread('src/image_recognition/photo.jpg')
    cv2.imshow("Input Image", img)
    contours = get_sudoku_board_from_image(img)
    result, location = adjust_for_perspective(img, contours)
    print(result, location)
    cv2.imshow("Board", result)
    cv2.waitKey() ### only for dev purposes




