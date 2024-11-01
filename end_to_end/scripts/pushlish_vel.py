import time
import argparse
import subprocess
import os
from os.path import join
import threading
import numpy as np
import rospy
import rospkg
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  # 导入 Float64 消息类型

def main():
    rospy.init_node('velocity_listener', anonymous=True)

    # 创建一个发布者，将线速度发布到 /linear_velocity 话题
    velocity_pub = rospy.Publisher('/linear_velocity', Odometry, queue_size=10)

    rate = rospy.Rate(10)  # 设置发布频率为10Hz
    while True:
      while not rospy.is_shutdown():
        try:
            data = rospy.wait_for_message('/jackal_velocity_controller/odom', Odometry, timeout=1)
            vel = [data.twist.twist.linear.x , data.twist.twist.angular.z]
            print ("the current vel is ", vel)
            velocity_pub.publish(data)
        except rospy.ROSException:
            print("no message received!!!!")
            break
       
        rate.sleep()  # 控制循环频率

if __name__ == '__main__':
    main()