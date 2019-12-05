#!/usr/bin/python2.7
# import roslib
import rospy
import sys
import cv2
import os
from cv_bridge import CvBridge, CvBridgeError
# Ros Messages
from sensor_msgs.msg import Image
import tensorflow as tf

sys.path.append("/data1/study/lanenet-ros")

from config import global_config
from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess

# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

CFG = global_config.cfg
weights_path = "../model/tuRVD_lanenet_vgg_2019-09-05-21-38-41.ckpt-84002"  # set the weights_path


class lanenet_processer:
    # member

    def __init__(self):

        # self.picProcesser = rospy.Subscriber("/imagetopic", CompressedImage, callback=self.callback, queue_size=10)
        self.picProcesser = rospy.Subscriber("/camera/image_color", Image, callback=self.callback,
                                             queue_size=10)  # set topic name
        self.result_puber = rospy.Publisher("/lanenet_result", Image, queue_size=10)  # output topic

        self.bridge = CvBridge()
        self.init_lanenet()
        self.image_count = 0

    def init_lanenet(self):
        """
        initialize the tensorflow model.
        """

        self.input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')
        net = lanenet.LaneNet(phase='test', net_flag='vgg')
        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='lanenet_model')
        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor()
        # load model
        saver = tf.train.Saver()
        #
        # Set sess configuration
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.TEST.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = CFG.TRAIN.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.Session(config=sess_config)
        saver.restore(sess=self.sess, save_path=weights_path)
        rospy.loginfo("init finish")

    def callback(self, image_msg):
        """Callback function of subscribed topic.
        Here images get converted and features detected"""
        rospy.loginfo("callback once")
        # process pre 30 frame
        if self.image_count >= 3:
            self.image_count = 0
        else:
            self.image_count = self.image_count + 1
            return
        try:
            rospy.loginfo("Process once")
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")

            # image_vis = cv_image.copy()
            image_vis = cv2.resize(cv_image, (1280,720), interpolation=cv2.INTER_LINEAR)
            image = cv2.resize(cv_image, (512, 256), interpolation=cv2.INTER_LINEAR)
            image = image / 127.5 - 1.0

            mask_image = self.inference_net(image, image_vis)
            output_result = self.bridge.cv2_to_imgmsg(mask_image, "bgr8")
            ##show the result image
            # cv2.imshow("test", postprocess_result['source_image'])
            # cv2.imshow("test", binary_seg_image[0] * 255)
            # cv2.waitKey(0)
            try:
                self.result_puber.publish(output_result)
            except CvBridgeError as e:
                print(e)
            rospy.loginfo("get a image with size %f", cv_image.size)
        except CvBridgeError as e:
            print e

    def inference_net(self, image, ori_image):
        binary_seg_image, instance_seg_image = self.sess.run(
            [self.binary_seg_ret, self.instance_seg_ret],
            feed_dict={self.input_tensor: [image]}
        )

        postprocess_result = self.postprocessor.postprocess(
            binary_seg_result=binary_seg_image[0],
            instance_seg_result=instance_seg_image[0],
            source_image=ori_image
        )
        # mask_image = postprocess_result
        # mask_image = cv2.resize(mask_image, (ori_image.shape[1],
        #                                      ori_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        # mask_image = cv2.addWeighted(ori_image, 0.6, mask_image, 5.0, 0)
        return postprocess_result['source_image']


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
