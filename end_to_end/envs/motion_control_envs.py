from gym.spaces import Box
import numpy as np

try:  # make sure to create a fake environment without ros installed
    import rospy
    from geometry_msgs.msg import Twist
except ModuleNotFoundError:
    pass

from envs.jackal_gazebo_envs import JackalGazebo, JackalGazeboLaser

class MotionControlContinuous(JackalGazebo):
    def __init__(self, min_v=-1, max_v=2, min_w=-3.14, max_w=3.14, **kwargs):
        self.action_dim = 2
        super().__init__(**kwargs)
        # if self.init_sim:
        self._cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        self.range_dict = RANGE_DICT = {
            "linear_velocity": [min_v, max_v],
            "angular_velocity": [min_w, max_w],
        }
        self.action_space = Box(
            low=np.array([RANGE_DICT["linear_velocity"][0], RANGE_DICT["angular_velocity"][0]]),
            high=np.array([RANGE_DICT["linear_velocity"][1], RANGE_DICT["angular_velocity"][1]]),
            dtype=np.float32
        )

    def reset(self):
        """reset the environment without setting the goal
        set_goal is replaced with make_plan
        """
        self.step_count = 0
        self.collision_count = 0
        # Reset robot in odom frame clear_costmap
        self.gazebo_sim.reset()
        self.start_time = self.current_time = rospy.get_time()
        pos, psi = self._get_pos_psi()
        
        self.gazebo_sim.unpause()
        obs,flage = self._get_observation(pos, psi, np.array([0, 0]))
        self.gazebo_sim.pause()
        
        goal_pos = np.array([self.world_frame_goal[0] - pos.x, self.world_frame_goal[1] - pos.y])
        self.last_goal_pos = goal_pos
        return obs

    def _take_action(self, action):
        robot_range=0.2
        scale = np.sqrt((0.254 ) **2 + (0.215 )**2)
        max_action_0 = 0.5
        max_action_1 = np.pi/2
        v_max = 2.0
        w_max = np.pi/2
        r_test = scale
        r_meta = robot_range
        v_m = (action[0]+1.0)*max_action_0/2
        w_m = action[1]*max_action_1
        '''
        calculation of the velocity of the tested robot based on the equations in our paper
        '''
        v_ideal = v_m*r_test/r_meta
        w_ideal = w_m
        v_max = v_max
        w_max = w_max
        rho_ideal = v_ideal/w_ideal
        if (v_max/w_max)<=abs(rho_ideal):
            v_test = np.minimum(v_max,v_ideal)
            w_test = v_test/rho_ideal
        else:
            w_test = np.minimum(abs(w_ideal),w_max)*np.sign(w_ideal)
            v_test = w_test*rho_ideal
        linear_speed = v_test    
        angular_speed =  w_test
        cmd_vel_value = Twist()
        cmd_vel_value.linear.x =  linear_speed 
        cmd_vel_value.angular.z =  angular_speed
        data = [linear_speed, angular_speed]
        print("the cmd vel is ", data)
        self.gazebo_sim.unpause()
        self._cmd_vel_pub.publish(cmd_vel_value)
        super()._take_action(action)  # this will wait util next time step
        self.gazebo_sim.pause()
        return [linear_speed,angular_speed]


class MotionControlContinuousLaser(MotionControlContinuous, JackalGazeboLaser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)