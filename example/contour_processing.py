from cv2_detection import MultipleProcessor, Processor
from cv2_detection.object_detection import Contour as AbstractContour


class Area(Processor):
    
    def __init__(self, min_area=500, max_area=8000):
        self.min_area = min_area
        self.max_area = max_area
    
    def is_correct_contour(self, contour: AbstractContour) -> bool:
        if self.min_area < contour.area < self.max_area:
            return True
        return False


class RectangularShape(Processor):

    def __init__(self, min_wh=0.35, max_wh=2.5):
        self.min_wh = min_wh
        self.max_wh = max_wh

    def is_correct_contour(self, contour: AbstractContour) -> bool:
        _, _, w, h = contour.bounding_rect
        if self.min_wh <= w / h <= self.max_wh:
            return True
        return False


class SquareShape(RectangularShape):

    def __init__(self):
        super().__init__(2/3, 3/2)


class LocatedInCenter(Processor):

    def __init__(self, left_x=0.2, right_x=0.8, down_y=0.2, up_y=0.8):
        self.left_x = left_x
        self.right_x = right_x
        self.up_y = up_y
        self.down_y = down_y

    def is_correct_contour(self, contour: AbstractContour) -> bool:
        x, y, w, h = contour.bounding_rect
        x, y = x + w // 2, y + h // 2
        if (
            self.left_x <= x / self.width <= self.right_x and
            self.down_y <= 1 - (y / self.height) <= self.up_y
        ):
            return True
        return False


class Sorter(Processor):
    
    def get_contours(self, contours: list[AbstractContour], height: int | float = 0, width: int | float = 0) -> list[AbstractContour]:
        return sorted(contours, key=lambda contour: -contour.area)


class SecondPointProcessor(MultipleProcessor):

    def __init__(self) -> None:
        processors = [SquareShape(), LocatedInCenter(), Sorter()]
        super().__init__(processors)


class FirstPointProcessor(MultipleProcessor):
    
    def __init__(self) -> None:
        processors = [RectangularShape(), Sorter()]
        super().__init__(processors)


