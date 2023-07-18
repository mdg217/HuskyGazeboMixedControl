import rospy
from geometry_msgs.msg import Twist

print("Ros topic publisher")

def mover():
    pub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist, queue_size=1)
    rospy.init_node('mover', anonymous=True)
    
    move_cmd = Twist()
    move_cmd.linear.x = 1.0
    move_cmd.angular.z = 0.2

    now = rospy.Time.now()
    rate = rospy.Rate(10)

    while rospy.Time.now() < now + rospy.Duration.from_sec(1010000):
        pub.publish(move_cmd)
        rate.sleep()
 
if __name__ == '__main__':
    try:
        mover()
    except rospy.ROSInterruptException:
        pass