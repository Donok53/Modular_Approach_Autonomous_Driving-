#!/usr/bin/python3
import math

import rospy
from dynamic_window_approach.msg import TrackedObjectArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (float(q.w) * float(q.z) + float(q.x) * float(q.y))
    cosy_cosp = 1.0 - 2.0 * (float(q.y) * float(q.y) + float(q.z) * float(q.z))
    return math.atan2(siny_cosp, cosy_cosp)


class TrackedObjectsMarkerViz:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/perception/tracked_objects")
        self.output_topic = rospy.get_param("~output_topic", "/perception/tracked_objects_markers")
        self.frame_id_override = str(rospy.get_param("~frame_id", "")).strip()
        self.marker_lifetime_s = max(0.0, float(rospy.get_param("~marker_lifetime_s", 0.6)))
        self.box_alpha = max(0.0, min(1.0, float(rospy.get_param("~box_alpha", 0.45))))
        self.text_alpha = max(0.0, min(1.0, float(rospy.get_param("~text_alpha", 0.95))))
        self.text_height_m = max(0.05, float(rospy.get_param("~text_height_m", 0.35)))
        self.arrow_width_m = max(0.01, float(rospy.get_param("~arrow_width_m", 0.08)))
        self.arrow_head_width_m = max(0.02, float(rospy.get_param("~arrow_head_width_m", 0.14)))
        self.arrow_scale_s = max(0.1, float(rospy.get_param("~arrow_scale_s", 0.8)))
        self.z_offset_m = float(rospy.get_param("~z_offset_m", 0.4))
        self.show_labels = bool(rospy.get_param("~show_labels", False))
        self.show_velocity = bool(rospy.get_param("~show_velocity", False))
        self.box_color = (
            max(0.0, min(1.0, float(rospy.get_param("~box_color_r", 0.98)))),
            max(0.0, min(1.0, float(rospy.get_param("~box_color_g", 0.82)))),
            max(0.0, min(1.0, float(rospy.get_param("~box_color_b", 0.20)))),
        )
        self.odom_topic = rospy.get_param("~odom_topic", "/lio_localizer/odometry/optimization")
        self.plan_range_m = max(0.0, float(rospy.get_param("~plan_range_m", 25.0)))
        self.perception_range_m = max(
            self.plan_range_m, float(rospy.get_param("~perception_range_m", 45.0))
        )
        self.plan_color = (
            max(0.0, min(1.0, float(rospy.get_param("~plan_color_r", 0.95)))),
            max(0.0, min(1.0, float(rospy.get_param("~plan_color_g", 0.10)))),
            max(0.0, min(1.0, float(rospy.get_param("~plan_color_b", 0.08)))),
        )
        self.perception_color = (
            max(0.0, min(1.0, float(rospy.get_param("~perception_color_r", 0.05)))),
            max(0.0, min(1.0, float(rospy.get_param("~perception_color_g", 0.05)))),
            max(0.0, min(1.0, float(rospy.get_param("~perception_color_b", 0.05)))),
        )
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.have_odom = False

        self.pub = rospy.Publisher(self.output_topic, MarkerArray, queue_size=2)
        self.sub = rospy.Subscriber(self.input_topic, TrackedObjectArray, self.callback, queue_size=5)
        self.sub_odom = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=20)

        rospy.loginfo(
            "tracked_objects_marker_viz started | in=%s out=%s odom=%s labels=%s velocity=%s plan=%.1fm perception=%.1fm plan_color=(%.2f, %.2f, %.2f) perception_color=(%.2f, %.2f, %.2f)",
            self.input_topic,
            self.output_topic,
            self.odom_topic,
            "on" if self.show_labels else "off",
            "on" if self.show_velocity else "off",
            self.plan_range_m,
            self.perception_range_m,
            self.plan_color[0],
            self.plan_color[1],
            self.plan_color[2],
            self.perception_color[0],
            self.perception_color[1],
            self.perception_color[2],
        )

    def odom_callback(self, msg):
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.have_odom = True

    def _marker_color_for_object(self, obj):
        if not self.have_odom:
            return self.box_color
        dx = float(obj.pose.position.x) - self.odom_x
        dy = float(obj.pose.position.y) - self.odom_y
        dist = math.hypot(dx, dy)
        if dist <= self.plan_range_m:
            return self.plan_color
        if dist <= self.perception_range_m:
            return self.perception_color
        return self.box_color

    def callback(self, msg):
        frame_id = self.frame_id_override or msg.header.frame_id or "map"
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        lifetime = rospy.Duration(self.marker_lifetime_s)

        marker_array = MarkerArray()
        delete_all = Marker()
        delete_all.header.stamp = stamp
        delete_all.header.frame_id = frame_id
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        marker_id = 0
        for obj in msg.objects:
            color = self._marker_color_for_object(obj)
            size_x = max(0.10, abs(float(obj.size.x)))
            size_y = max(0.10, abs(float(obj.size.y)))
            size_z = max(0.10, abs(float(obj.size.z)))

            box = Marker()
            box.header.stamp = stamp
            box.header.frame_id = frame_id
            box.ns = "boxes"
            box.id = marker_id
            marker_id += 1
            box.type = Marker.CUBE
            box.action = Marker.ADD
            box.pose = obj.pose
            box.pose.position.z = float(obj.pose.position.z) + 0.5 * size_z + self.z_offset_m
            box.scale.x = size_x
            box.scale.y = size_y
            box.scale.z = size_z
            box.color.r = color[0]
            box.color.g = color[1]
            box.color.b = color[2]
            box.color.a = self.box_alpha
            box.lifetime = lifetime
            marker_array.markers.append(box)

            if self.show_labels:
                text = Marker()
                text.header.stamp = stamp
                text.header.frame_id = frame_id
                text.ns = "labels"
                text.id = marker_id
                marker_id += 1
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = float(obj.pose.position.x)
                text.pose.position.y = float(obj.pose.position.y)
                text.pose.position.z = float(obj.pose.position.z) + size_z + self.z_offset_m + 0.25
                text.pose.orientation.w = 1.0
                text.scale.z = self.text_height_m
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = self.text_alpha
                text.text = "{} #{:d}".format(str(obj.label or "obj"), int(obj.id))
                text.lifetime = lifetime
                marker_array.markers.append(text)

            if self.show_velocity:
                vx = float(obj.twist.linear.x)
                vy = float(obj.twist.linear.y)
                if math.hypot(vx, vy) > 0.02:
                    arrow = Marker()
                    arrow.header.stamp = stamp
                    arrow.header.frame_id = frame_id
                    arrow.ns = "velocity"
                    arrow.id = marker_id
                    marker_id += 1
                    arrow.type = Marker.ARROW
                    arrow.action = Marker.ADD
                    arrow.scale.x = self.arrow_width_m
                    arrow.scale.y = self.arrow_head_width_m
                    arrow.scale.z = self.arrow_head_width_m
                    arrow.color.r = color[0]
                    arrow.color.g = color[1]
                    arrow.color.b = color[2]
                    arrow.color.a = 0.95
                    start = Point()
                    start.x = float(obj.pose.position.x)
                    start.y = float(obj.pose.position.y)
                    start.z = float(obj.pose.position.z) + size_z + self.z_offset_m + 0.05
                    end = Point()
                    end.x = start.x + vx * self.arrow_scale_s
                    end.y = start.y + vy * self.arrow_scale_s
                    end.z = start.z
                    arrow.points = [start, end]
                    arrow.lifetime = lifetime
                    marker_array.markers.append(arrow)

        self.pub.publish(marker_array)


def main():
    rospy.init_node("tracked_objects_marker_viz", anonymous=False)
    TrackedObjectsMarkerViz()
    rospy.spin()


if __name__ == "__main__":
    main()
