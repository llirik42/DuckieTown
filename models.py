from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv

import numpy as np


class RoadStretch:
    __length: float
    __angle: float

    def __init__(self, length: float, angle_rad: float):
        self.__length = length
        
        self.__angle = angle_rad

    @property
    def length(self) -> float:
        return self.__length

    @property
    def angle_deg(self) -> float:
        return np.rad2deg(self.__angle)

    @property
    def angle_rad(self) -> float:
        return self.__angle


class GraphNode:
    def __init__(self, x_center: float, y_center: float):
        self.__x = x_center
        self.__y = y_center
        self.__neighbors = []

    def add_neighbor(self, neighbor, distance):
        pass




class Graph:
    pass


class Driver:
    _action: np.array
    __wheel_distance: float
    __min_rad: float

    def __init__(self):
        self._action = np.array([0.0, 0.0])
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

    def get_action(self) -> np.array:
        self._update_action_by_control()
        self.__adjust_action()

        return self._action


class ManualDriver(Driver):
    __key_handler: key.KeyStateHandler

    def __init__(self, key_handler: key.KeyStateHandler):
        super().__init__()

        self.__key_handler = key_handler
        
    def _update_action_by_control(self):
        self._action = np.array([0.0, 0.0])
        
        is_moving_up = self.__key_handler[key.UP] or self.__key_handler[key.W]
        is_moving_down = self.__key_handler[key.DOWN] or self.__key_handler[key.S]
        is_moving_left = self.__key_handler[key.LEFT] or self.__key_handler[key.A]
        is_moving_right = self.__key_handler[key.RIGHT] or self.__key_handler[key.D]

        if is_moving_up:
            self._action += np.array([0.44, 0.0])
        if is_moving_down:
            self._action -= np.array([0.44, 0])
        if is_moving_left:
            self._action += np.array([0, 1])
        if is_moving_right:
            self._action -= np.array([0, 1])
   

class AutoDriver(Driver):
    def __init__(self, environment: DuckietownEnv):
        super().__init__()

        self.__environment = environment

    @staticmethod
    def get_independence(power: float, number: float):
        return abs(number) ** power * np.sign(number)

    def _update_action_by_control(self):
        self._action = np.array([0.0, 0.0])
        
        lane_pose = self.__environment.get_lane_pos2(self.__environment.cur_pos, self.__environment.cur_angle)
        distance_to_road_center = lane_pose.dist + 0.14
        angle_to_road = lane_pose.angle_rad

        max_speed = 0.3  # максимальная скорость (при прямолинейном движении)
        difference_of_angles = 1 - 0.5 * abs(angle_to_road) / np.pi  # насколько нужно повернуть, чтоб смотреть на дорогу прямо
        speed = max_speed * self.get_independence(number=difference_of_angles, power=20)  # зависимость скорости движения вперёд, от угла поворота

        k = 5 # коэффициент перед расстоянием до центра дороги при вычислении поворота
        rotation = self.get_independence(number=angle_to_road, power=0.5)  # вычисление поворота в зависимости от угла поворота
        rotation += k * self.get_independence(number=distance_to_road_center, power=1)  # вычисление поворота в зависимости от расстояния до центра дороги

        print(f'Угол = {round(angle_to_road, 2)} Расстояние до дороги = {round(distance_to_road_center, 2)}')

        self._action = np.array([speed, rotation]) 

    # def __adjust_action(self):
        # pass