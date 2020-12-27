from models import *

import cv2


image_size = 800


node1 = GraphNode(x_center=1, y_center=4)

node2 = GraphNode(x_center=2, y_center=6)

node3 = GraphNode(x_center=-3, y_center=-10)
node4 = GraphNode(x_center=-4, y_center=11)

nodes = [node1, node2, node3, node4]

min_x = None
max_x = None
min_y = None
max_y = None

for node in nodes:
    current_x = node.x
    current_y = node.y

    if min_x is None:
        min_x = current_x
    if max_x is None:
        max_x = current_x

    if min_y is None:
        min_y = current_y
    if max_y is None:
        max_y = current_y

    if current_x > max_x:
        max_x = current_x
    if current_x < min_x:
        min_x = current_x

    if current_y > max_y:
        max_y = current_y
    if current_y < min_y:
        min_y = current_y

x_delta = max_x - min_x
y_delta = max_y - min_y

image = np.zeros((image_size, image_size, 3), np.uint8)

center_coordinates = (400, 400)

R0 = 200
radius = R0 // 2
color = (255, 255, 255)
thickness = 2 * radius

cv2.circle(image, center_coordinates, radius, color, thickness)

print(type(image))

cv2.imshow('image', image)

cv2.waitKey(0)
