import numpy as np
from robot import Simple_Manipulator as Robot

def M5(robot: Robot, path: np.array) -> np.array:
    """Smooth the given path

    Parameters
    ----------
    robot : Robot
        our robot object
    path : np.array
        Nx4 numpy array containing a collision-free path between q_start and q_goal

    Returns
    -------
    np.array
        Nx4 numpy array containing a smoothed version of the
        input path, where some unnecessary intermediate
        waypoints may have been removed
    """

    #student work start here
    start_index = 0
    while start_index < len(path) - 1:
        #loop over nodes in reverse order
        for end_index in reversed(range(start_index + 2 ,len(path))): # start index + 1 would not have any nodes to remove
            #try to remove intermediate points - check next nearest node if can't remove
            if robot.check_edge(path[start_index], path[end_index]):
                # path = path[:start_index + 1] + path[end_index : ] #if list
                path = np.vstack((path[:start_index + 1], path[end_index : ]))
                break
        #advance search node 1 forward
        start_index += 1

    return path