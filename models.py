from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from typing import List, Dict, Tuple

import numpy

import cv2


class XY:
    __x: float
    __y: float

    def __init__(self, x: float, y: float):
        self.__x = x
        self.__y = y

    @property
    def x(self) -> float:
        return self.__x

    @property
    def y(self) -> float:
        return self.__y

    @property
    def x_y(self) -> Tuple[float, float]:
        return self.__x, self.__y

    @property
    def int_x_y(self) -> Tuple[int, int]:
        return int(self.__x), int(self.__y)

class GraphNode:
    __center: XY
    __neighbors: List[Dict]

    def __init__(self, center: XY):
        self.__center = center
        self.__neighbors = []

    @property
    def x(self) -> float:
        return self.__center.x

    @property
    def y(self) -> float:
        return self.__center.y

    @property
    def center(self) -> XY:
        return self.__center

    @staticmethod
    def get_distance(point1: XY, point2: XY) -> float:
        return numpy.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def add_neighbor(self, neighbor_node):
        neighbor = {
            'distance': self.get_distance(point1=self.__center, point2=neighbor_node.__center),
            'node': neighbor_node,
        }

        self.__neighbors.append(neighbor)

    def print_neighbors(self):
        print(self.__neighbors)

    def __repr__(self):
        return f'Node:({self.x};{self.y})'


class Graph:
    __nodes: List[GraphNode]

    def __init__(self):
        self.__nodes = []

    @property
    def nodes(self):
        return self.__nodes

    def add_node(self, node: GraphNode):
        self.__nodes.append(node)

    def __link_nodes(self, index1, index2):
        node1 = self.__nodes[index1]
        node2 = self.__nodes[index2]

        node1.add_neighbor(neighbor_node=node2)
        node2.add_neighbor(neighbor_node=node1)

    def link_last_two_nodes(self):
        number_of_nodes = len(self.__nodes)

        self.__link_nodes(number_of_nodes-1, number_of_nodes-2)

    def __repr__(self):
        return f'{self.__nodes}'


class GraphDrawer:
    __current_image: numpy.ndarray

    def __init__(self, save_aspect_ratio: bool=True, image_max_size: int=800, color: Tuple[int, int, int]=(255, 255, 255),
                 background_color: Tuple[int, int, int]=(0, 0, 0), nodes_radius: int=10):
        self.__nodes_radius = nodes_radius
        self.__save_aspect_ratio = save_aspect_ratio
        self.__max_size = image_max_size
        self.__background_color = background_color
        self.__color = color

    @staticmethod
    def __scale(value: float, left_border: float, right_border: float, max_value: float) -> int:
        return int((value - left_border) * max_value / (right_border - left_border))

    def __draw_circle(self, image: numpy.ndarray, center: Tuple[int, int]):
        radius = self.__nodes_radius // 2
        thickness = 2 * radius

        cv2.circle(image, center, radius, self.__color, thickness)

    @staticmethod
    def get_min_and_max_node_x_y(nodes: List[GraphNode]) -> Tuple[float, float, float, float]:
        min_x = nodes[0].x
        max_x = min_x

        min_y = nodes[0].y
        max_y = min_y

        for node in nodes:
            current_x = node.x
            current_y = node.y

            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)

            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)

        return min_x, max_x, min_y, max_y

    def __get_image_dimensions(self, min_x, max_x, min_y, max_y) -> Tuple[int, int]:
        x_delta = max_x - min_x
        y_delta = max_y - min_y

        pixels_per_delta_unit = self.__max_size / max(abs(x_delta), abs(y_delta))
        x_size = int(x_delta * pixels_per_delta_unit)
        y_size = int(y_delta * pixels_per_delta_unit)

        dimensions = (x_size, y_size)

        return dimensions

    def draw_nodes(self, image: numpy.ndarray, nodes: List[GraphNode]):
        for node in nodes:
            current_x = node.x
            current_y = node.y

            x_center = self.__scale(value=current_x, left_border=)

            self.__draw_circle(image=image)

    def get_image(self, graph: Graph) -> numpy.ndarray:
        nodes = graph.nodes

        min_x, max_x, min_y, max_y = self.get_min_and_max_node_x_y(nodes=nodes)
        image_x_size, image_y_size = self.__get_image_dimensions(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

        image = numpy.zeros((image_x_size, image_y_size, 3), numpy.uint8)

        self.draw_nodes(image=image, nodes=nodes)

        return image



class Driver:
    _action: numpy.array
    __wheel_distance: float
    __min_rad: float

    def __init__(self):
        self._action = numpy.array([0.0, 0.0])
        self.__wheel_distance = 0.102
        self.__min_rad = 0.08

    def _update_action_by_control(self):
        pass

    def __adjust_action(self):
        v1 = self._action[0]
        v2 = self._action[1]
        # Limit radius of curvature
        if v1 == 0 or abs(v2 / v1) > (self.__min_rad + self.__wheel_distance / 2.0) / (self.__min_rad - self.__wheel_distance / 2.0):
            # adjust velocities evenly such that condition is fulfilled
            delta_v = (v2 - v1) / 2 - self.__wheel_distance / (4 * self.__min_rad) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v

        self._action[0] = v1
        self._action[1] = v2

    def get_action(self) -> numpy.array:
        self._update_action_by_control()
        self.__adjust_action()

        return self._action


class ManualDriver(Driver):
    __key_handler: key.KeyStateHandler

    def __init__(self, key_handler: key.KeyStateHandler):
        super().__init__()

        self.__key_handler = key_handler
        
    def _update_action_by_control(self):
        self._action = numpy.array([0.0, 0.0])
        
        is_moving_up = self.__key_handler[key.UP] or self.__key_handler[key.W]
        is_moving_down = self.__key_handler[key.DOWN] or self.__key_handler[key.S]
        is_moving_left = self.__key_handler[key.LEFT] or self.__key_handler[key.A]
        is_moving_right = self.__key_handler[key.RIGHT] or self.__key_handler[key.D]

        if is_moving_up:
            self._action += numpy.array([0.44, 0.0])
        if is_moving_down:
            self._action -= numpy.array([0.44, 0])
        if is_moving_left:
            self._action += numpy.array([0, 1])
        if is_moving_right:
            self._action -= numpy.array([0, 1])
   

class AutoDriver(Driver):
    __environment: DuckietownEnv

    def __init__(self, environment: DuckietownEnv):
        super().__init__()

        self.__environment = environment

    @staticmethod
    def get_independence(power: float, number: float):
        return abs(number) ** power * numpy.sign(number)

    @staticmethod
    def round(value: float) -> float:
        return round(value, 2)

    def _update_action_by_control(self):
        self._action = numpy.array([0.0, 0.0])
        
        lane_pose = self.__environment.get_lane_pos2(self.__environment.cur_pos, self.__environment.cur_angle)
        distance_to_road_center = lane_pose.dist + 0.14
        angle_to_road = lane_pose.angle_rad

        max_speed = 0.2  # максимальная скорость (при прямолинейном движении)
        difference_of_angles = 1 - 0.5 * abs(angle_to_road) / numpy.pi  # насколько нужно повернуть, чтоб смотреть на дорогу прямо
        speed = max_speed * self.get_independence(number=difference_of_angles, power=10)  # зависимость скорости движения вперёд, от угла поворота

        k = 4 # коэффициент перед расстоянием до центра дороги при вычислении поворота
        rotation = self.get_independence(number=angle_to_road, power=0.4)  # вычисление поворота в зависимости от угла поворота
        rotation += k * self.get_independence(number=distance_to_road_center, power=1.05)  # вычисление поворота в зависимости от расстояния до центра дороги

        # print(f'Скорость = {self.round(speed)}  поворот = {self.round(rotation)}  расстояние = {self.round(distance_to_road_center)}  угол = {self.round(angle_to_road)}')

        self._action = numpy.array([speed, rotation])
