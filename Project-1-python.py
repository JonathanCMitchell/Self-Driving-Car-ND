import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from helpers import grayscale
from helpers import canny
from helpers import gaussian_blur
from helpers import region_of_interest
from helpers import draw_lines
from helpers import hough_lines
from helpers import weighted_img
# no apply_canny, no apply_gray
lastCoordinatesLeft_max = (0, 0)
lastCoordinatesLeft_min = (0, 0)
lastCoordinatesRight_min = (0, 0)
lastCoordinatesRight_max = (0, 0)

def process_image(image):
    gray = grayscale(image)
    blur_gray = apply_gaussian_blur(gray)
    edges = apply_canny(blur_gray)
    masked_image = apply_mask(edges, image, gray)
    hough_image, lines = apply_hough(masked_image)
    lines_edges_weighted = apply_lines_edges_weighted(image, edges, hough_image)
    return lines_edges_weighted

def apply_gaussian_blur(gray, kernel_size = 5):
    return gaussian_blur(gray, kernel_size)

def apply_mask(edges, image, gray):
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    apex_left = [400, 355]
    apex_right = [imshape[1] - 380, 355]
    bottom_left = [175, imshape[0]]
    bottom_right = [imshape[1] - 39,imshape[0]]
    vertices = np.array([[(bottom_left[0],bottom_left[1]), \
                          (apex_left[0],apex_left[1]), \
                          (apex_right[0], apex_right[1]), \
                          (bottom_right[0], bottom_right[1])]], dtype=np.int32)

    (masked_image, mask) = region_of_interest(edges, vertices)
    return masked_image


def apply_hough(masked_image):
    rho = 1 
    theta = np.pi/180 
    threshold = 50
    min_line_len = 50
    max_line_gap = 200

    # Create a line image with hough lines on it
    hough_image, lines = hough_lines(masked_image, rho, theta, threshold, min_line_len, max_line_gap)
    return hough_image, lines

def apply_lines_edges_weighted(image, edges, hough_image):
    lines_edges_weighted = weighted_img(hough_image, image)
    return lines_edges_weighted

