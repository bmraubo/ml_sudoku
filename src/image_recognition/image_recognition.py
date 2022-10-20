import cv2
import imutils
# import numpy
# from tensorflow import keras
# from keras.models import load_model

def get_sudoku_board_from_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    newimg = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg)

if __name__ == "__main__":
    img = cv2.imread('src/image_recognition/easypuzzle.png')
    cv2.imshow("Input Image", img)
    get_sudoku_board_from_image(img)
    cv2.waitKey()




