import cv2
import numpy
import imutils


class BoardReader:
    bilateral_filter_settings = {
        "filter_size": 13,
        "sigma_color": 20,
        "sigma_space": 20,
    }

    number_block_image_size = (300, 300)

    def __init__(self, image_file_location: str) -> None:
        self.image_file_location = image_file_location

    def extract_blocks_from_image(self):
        self.image = cv2.imread(self.image_file_location)
        edges, kernel = self.prepare_image()
        self.contours = self.identify_contours(edges, kernel)
        self.board, self.location = self.adjust_for_perspective()
        self.number_blocks = self.identify_number_blocks()
        return self.number_blocks

    def prepare_image(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        bilateral_filter = cv2.bilateralFilter(
            gray,
            self.bilateral_filter_settings["filter_size"],
            self.bilateral_filter_settings["sigma_color"],
            self.bilateral_filter_settings["sigma_space"],
        )
        edges = cv2.Canny(bilateral_filter, 30, 180)
        kernel = numpy.ones((3, 3), numpy.uint8)
        return edges, kernel

    def identify_contours(self, edges, kernel):
        edges = cv2.dilate(edges, kernel, iterations=1)
        keypoints = cv2.findContours(
            edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        return imutils.grab_contours(keypoints)

    def adjust_for_perspective(self):
        def identify_rectangular_contours(contours):
            for contour in contours:
                approx = cv2.approxPolyDP(contour, 15, True)
                if len(approx) == 4:
                    return approx

        def sort_contours(contours):
            return sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        def apply_perspective_adjustment(location, height=900, width=900):
            source = numpy.float32([location[0], location[3], location[1], location[2]])
            destination = numpy.float32(
                [[0, 0], [width, 0], [0, height], [width, height]]
            )
            matrix = cv2.getPerspectiveTransform(source, destination)
            result = cv2.warpPerspective(self.image, matrix, (width, height))
            return result

        try:
            sorted_contours = sort_contours(self.contours)
            location = identify_rectangular_contours(sorted_contours)
            result = apply_perspective_adjustment(location)
            return result, location
        except:
            raise Exception("board not found")

    def identify_number_blocks(self):
        rows = numpy.vsplit(self.board, 9)
        blocks = []
        for row in rows:
            boxes = numpy.hsplit(row, 9)
            for box in boxes:
                block = cv2.resize(box, self.number_block_image_size) / 255.0
                blocks.append(block)
        return blocks
