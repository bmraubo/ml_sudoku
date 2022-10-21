import cv2
import time

from src.image_recognition.board_reader import BoardReader


def show_blocks(board_reader):
    block_number = 1
    for block in board_reader.number_blocks:
        cv2.imshow(f"block #{block_number}", block)
        cv2.waitKey(100)
        block_number += 1

def show_board(board_reader):
    cv2.imshow("board", board_reader.board)
    cv2.waitKey(400)

def show_original_image(board_reader):
    cv2.imshow("image", board_reader.image)
    cv2.waitKey(400)

def show_image_contours(board_reader):
    newimg = cv2.drawContours(board_reader.image.copy(), board_reader.contours, -1, (0, 255, 0), 3)
    cv2.imshow("Contour", newimg)
    cv2.waitKey(400)

test_file = "test/image_recognition/simple_test.jpeg"

board_reader = BoardReader(test_file)
board_reader.extract_blocks_from_image()

show_original_image(board_reader)
time.sleep(2)
show_image_contours(board_reader)
time.sleep(2)
show_board(board_reader)
time.sleep(2)
show_blocks(board_reader)
