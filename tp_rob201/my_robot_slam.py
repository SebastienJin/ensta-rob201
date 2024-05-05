"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, potential_field_control_tp5
from occupancy_grid import OccupancyGrid
from planner import Planner
from time import time


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0

        # Init SLAM object
        self._size_area = (1500, 800)
        self.occupancy_grid = OccupancyGrid(x_min=- self._size_area[0],
                                            x_max=self._size_area[0],
                                            y_min=- self._size_area[1],
                                            y_max=self._size_area[1],
                                            resolution=2)
        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

        # target index for a list of goals
        self.target = 0

    def control(self):
        """
        Main control function executed at each time step
        """
        return self.control_tp5()

    def control_tp1(self):
        """
        Control function for TP1
        """
        self.tiny_slam.compute()
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        """
        pose = self.odometer_values()
        # path following
        goal = np.array([[-400, 0, 0],[-400, 210, 0],[-105, 210, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)

        return command
    
    def control_tp3(self):
        """
        Control function for TP3
        """
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        
        pose = self.odometer_values()
        goal = np.array([[0, -400, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)

        return command
    
    def control_tp4(self):
        """
        Control function for TP4
        """
        # every 10 steps, localise
        self.counter += 1
        if self.counter % 10 == 0:
            self.tiny_slam.localise(self.lidar(), self.odometer_values())
        
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), pose)

        goal = np.array([[50, -200, 0],[-100, -400, 0],[-300, -450, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)
        
        return command
    
    def control_tp5(self):
        """
        Control function for TP5
        """
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), pose, True)

        if self.occupancy_grid.returning == 0:
            command = potential_field_control_tp5(self.lidar(), pose, self.occupancy_grid.goal)

            # if we are close to the goal, plan a path back to start
            if np.linalg.norm(self.tiny_slam.get_corrected_pose(self.odometer_values(), None) - self.occupancy_grid.goal) <= 5:

                # just reach the main goal, calculate a path first
                start = time()
                self.occupancy_grid.path = self.planner.plan(np.array([0, 0, 0]), self.tiny_slam.get_corrected_pose(self.odometer_values(), None) )
                print("Time of calculation : ", round(time()- start,3), " s.")

                # update basic variables
                self.occupancy_grid.returning = 1
                # self.occupancy_grid.index_back_to_start = 0

        # following the return path
        else:
            command = command = potential_field_control_tp5(self.lidar(), pose, self.occupancy_grid.goal, False)

            if np.linalg.norm(self.tiny_slam.get_corrected_pose(self.odometer_values(), None) - self.occupancy_grid.goal) <= 5:
                # follow by reseting the goal recursively
                n_step = 5
                if self.occupancy_grid.index_back_to_start < len(self.occupancy_grid.path) - n_step:
                    x_new_goal, y_new_goal = self.tiny_slam.grid.conv_map_to_world(
                        self.occupancy_grid.path[self.occupancy_grid.index_back_to_start][0],
                        self.occupancy_grid.path[self.occupancy_grid.index_back_to_start][1])
                    self.occupancy_grid.index_back_to_start += n_step
                    self.occupancy_grid.goal = np.array([x_new_goal, -y_new_goal, 0])
                else:
                    # final goal - (0,0,0)
                    self.occupancy_grid.goal = np.array([0, 0, 0])

        return command
