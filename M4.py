from M1 import M1
import numpy as np
from robot import Simple_Manipulator as Robot
import typing

def M4(robot: Robot, q_start: np.array, q_goal: np.array) -> typing.Tuple[np.array, bool]:
    """Implement RRT algorithm to find a path from q_start to q_goal

    Parameters
    ----------
    robot : Robot
        our robot object
    q_start : np.array
        1x4 numpy array denoting the start configuration
    q_goal : np.array
        1x4 numpy array denoting the goal configuration

    Returns
    -------
    typing.Tuple[np.array, bool]
        np.array:
            Nx4 numpy array containing a collision-free path between
            q_start and q_goal, if a path is found. The first row
            should be q_start, the final row should be q_goal.
        bool:
            Boolean denoting whether a path was found
    """

    # student work start here
    # student work start here
    path = []
    path_found = False

    #parameters
    max_iters = 1000000
    max_sep = 0.2
    
    #tree structure
    nodes = [q_start]
    parents = [0]
    iter_count = 0
    while iter_count < max_iters:
        iter_count += 1

        #get new random node
        if np.random.random() < 0.1: #bias the algorithm slightly toward the goal
            new_node = q_goal
        else:
            new_node = M1(robot.lower_lims, robot.upper_lims, 1).reshape(-1)

        #find closest node
        distances = np.sum( np.abs( np.array(nodes) - new_node ), axis = 1)
        closest_node_index = np.argsort(distances)[0]
        closest_node = nodes[closest_node_index]

        #check max dist between nodes
        dist = np.linalg.norm(new_node - closest_node)
        if dist > max_sep:
            #interpolate to move the node closer
            new_node = closest_node + (max_sep/dist) * (new_node - closest_node)
        
        #check for collisions
        if robot.check_edge(closest_node, new_node):
            #add to tree
            nodes.append(new_node)
            parents.append(closest_node_index)
            
            #check for complete
            if np.all(new_node == q_goal):
                path_found = True
                print("M4 -- Stopping condition reached")
                break

            #check near completion (not strictyl necessary)
            dist_to_goal = np.linalg.norm(new_node - q_goal)
            if dist_to_goal < max_sep:
                if robot.check_edge(new_node, q_goal):
                    #stop iterating
                    nodes.append(q_goal)
                    parents.append(len(nodes)-2)
                    path_found = True
                    print("M4 -- Stopping condition reached")
                    break
    
    #traverse up tree to get the path
    if path_found:
        idx = -1
        path_rev = [-1]
        while not path_rev[-1] == 0:
            idx = parents[idx]
            path_rev.append(idx)
        path = [nodes[i] for i in reversed(path_rev)]
    else:
        print("M4 -- Reached iteration limit")
    #ensure matches spec
    path = np.array(path)
                        
    return path, path_found