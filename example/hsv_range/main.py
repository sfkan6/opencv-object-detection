from cv2_detection import HSVThresher, Image


def main():
    image = Image.read('test.jpg')

    red_hsv_ranges = [
        [(0, 150, 30), (15, 255, 255)],
        [(175, 0, 30), (180, 255, 255)]
    ]
    threshold_image = HSVThresher(red_hsv_ranges).get_threshold_image(image)
    threshold_image.write('threshold_image.jpg')


if __name__ == '__main__':
    main()
