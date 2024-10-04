from cv2_detection.object_detection import Detector
from cv2_detection.object_detection.indication import Indicator, MultipleIndicator
from cv2_detection import Camera, Image, Thresher, HSVThresher, RectanglePainter, ContourPainter
from contour_processing import FirstPointProcessor, SecondPointProcessor
import cv2


def main():
    image = Image.read("test.jpg")

    hsv_ranges = [
        [(0, 150, 150), (10, 255, 255)],
        [(175, 0, 150), (180, 255, 255)],
    ]
    thresher = HSVThresher(hsv_ranges)


    threshold_image = thresher.get_finished_image(image)
    threshold_image.write("threshold.png")
    
    contours = thresher.get_contours(image)
    contour_image = image.get_image_by_bounding_rect(*contours[0].bounding_rect)

    rectangle_painter = RectanglePainter(color=(0, 255, 0), thickness=3)
    new_image = rectangle_painter.draw_all(image, contours)

    contour_image.write("contour-image.png")
    new_image.write("test-with-rect-contours.png")



def example():

    # Image
    # image = Camera().get_image()
    # image = Image(cv2.imread("test.jpg"))
    image = Image.read("test.jpg")


    # Thresher
    thresher = Thresher()

    hsv_ranges = [
        [(0, 150, 150), (10, 255, 255)],
        [(175, 0, 150), (180, 255, 255)],
    ]
    thresher = HSVThresher(hsv_ranges) 


    # Detector
    first_detector = Detector(thresher, FirstPointProcessor())
    second_detector = Detector(thresher, SecondPointProcessor())
    
    contours = first_detector.get_contours_by_image(image)


    # Painter
    contour_painter = ContourPainter(color=(0, 255, 0), thickness=3)
    rectangle_painter = ContourPainter(color=(0, 255, 0), thickness=3)

    new_image1 = contour_painter.draw(image, contours[0])
    new_image2 = contour_painter.draw_all(image, contours)
    new_image3 = rectangle_painter.draw_all(image, contours)


    # Indicator
    contour_indicator = Indicator(first_detector, contour_painter)
    rectangle_indicator = Indicator(second_detector, rectangle_painter)

    new_image1 = contour_indicator.display(image)
    new_image2 = rectangle_indicator.display(image)


    # MultipleIndicator
    multiple_indicator = MultipleIndicator([contour_indicator, rectangle_indicator])
    new_image = multiple_indicator.display(image)

    # cv2.imwrite("new-test.png", new_image._image)
    new_image.write("new-with-contours.png")



if __name__ == "__main__":
    main()
