"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner


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
        self._size_area = (800, 1200)
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
        return self.control_tp4()

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
        # goal = np.array([[-400, 200, 0],[-105, 200, 0],[-100, 480, 0],[-15, 550, 0]])
        goal = np.array([[-400, 0, 0],[-400, 210, 0],[-105, 210, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)

        return command
    
    def control_tp3(self):
        """
        Control function for TP3
        """
        self.tiny_slam.update_map(self.lidar(), self.odometer_values())
        
        pose = self.odometer_values()
        goal = np.array([[-400, 0, 0],[-400, 200, 0],[-105, 200, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)

        return command
    
    def control_tp4(self):
        """
        Control function for TP4
        """
        self.counter += 1
        
        self.tiny_slam.localise(self.lidar(), self.odometer_values())
        
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        self.tiny_slam.update_map(self.lidar(), pose)

        goal = np.array([[-400, 0, 0],[-400, 200, 0],[-105, 200, 0]])
        command, self.target = potential_field_control(self.lidar(), pose, goal, self.target)
        
        return command
