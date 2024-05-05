import numpy as np
import heapq
import itertools
from occupancy_grid import OccupancyGrid


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):

        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def plan(self, start, goal):
        """
        Compute a path using A*, recompute plan if start or goal change
        start : [x, y, theta] nparray, start pose in world coordinates (theta unused)
        goal : [x, y, theta] nparray, goal pose in world coordinates (theta unused)
        """
        # TODO for TP5

        # path = [start, goal]  # list of poses

        start = self.grid.conv_world_to_map(start[0], -start[1])
        goal  = self.grid.conv_world_to_map(goal[0], -goal[1])

        path = self.A_Star(start, goal)
        return path

    def explore_frontiers(self):
        """ Frontier based exploration """
        # frontier to reach for exploration
        goal = np.array([0, 0, 0])  
        return goal
    
    def get_neighbors(self, current_cell):
        """ Get the neighbors of a cell """
        directions = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]])
        neighbors = []
        for direction in directions:
            neighbor = (current_cell[0] + direction[0], current_cell[1] + direction[1])
            # in the map and not occupied
            if (0 <= neighbor[0] < self.grid.x_max_map) and (0 <= neighbor[1] < self.grid.y_max_map) and self.grid.occupancy_map[neighbor[0]][-neighbor[1]] < 0:
                neighbors.append(neighbor)
        return neighbors
    
    def heuristic(self, cell_1, cell_2):
        """ Compute the heuristic between two cells """
        return np.linalg.norm(np.array(cell_1) - np.array(cell_2))
    
    def reconstruct_path(self, cameFrom, current):
        """ Reconstruct the path """
        total_path = [current]
        while current in cameFrom.keys() and cameFrom[current] != None:
            current = cameFrom[current]
            total_path.append(current)
        return total_path
    
    def A_Star(self, start, goal):
        """ A* algorithm """
        # form: (Corresponding weight, position)
        openSet = [(self.heuristic(start,goal), start)]         

        # set of location already passed
        visited_nodes = set()
        
        # Dictionary used to record the position of the previous step
        cameFrom = {start: None}             
        
        # Dictionary used to record the gScore
        gScore= {start : 0} 

        # Dictionary used to record the fScore, f = g + h
        fScore = {start : self.heuristic(start, goal)} 
        
        while openSet:
            # position with the smallest fScore
            current = heapq.heappop(openSet)[1]
            
            # If we reach the goal, we reconstruct the path
            if current == goal:
                return self.reconstruct_path(cameFrom, current)
            
            visited_nodes.add(current)
            neighbors = self.get_neighbors(current)

            for neighbor in neighbors:
                
                    # Compute temporary gScore
                    tentative_gScore = gScore[current] + self.heuristic(current, neighbor)

                    # already been visited and gScore is bigger than now
                    if neighbor in visited_nodes and tentative_gScore >= gScore[neighbor]:
                        continue
                    
                    # update basic information
                    cameFrom[neighbor] = current                                 
                    gScore[neighbor] = tentative_gScore                          
                    fScore[neighbor] = tentative_gScore + self.heuristic(neighbor, goal) 
                    if ((fScore[neighbor], neighbor) not in openSet):
                        heapq.heappush(openSet, (fScore[neighbor], neighbor))

        return None
