#!/usr/bin/python2.7
import rospy
# import roslib
import cv2
import sys
from std_msgs.msg import String
from cv_bridge import CvBridge,CvBridgeError
# Ros Messages
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage


# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError


class lanenet_processer:
    def __init__(self):
        # self.picProcesser = rospy.Subscriber("/imagetopic", CompressedImage, callback=self.callback, queue_size=10)
        self.picProcesser = rospy.Subscriber("/camera/image_color", Image, callback=self.callback, queue_size=10)
        self.bridge = CvBridge()

    def callback(self, image_msg):
        """Callback function of subscribed topic.
        Here images get converted and features detected"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.loginfo("get a image with size %f", cv_image.size)
        except CvBridgeError as e:
            print e


def main(args):
    """Initializes and cleanup ros node"""
    ic = lanenet_processer()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"


if __name__ == '__main__':
    main(sys.argv)
