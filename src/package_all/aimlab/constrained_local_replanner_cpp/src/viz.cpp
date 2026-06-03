#include "constrained_local_replanner_cpp/viz.hpp"

#include <sstream>

#include <std_msgs/String.h>

namespace clr {

namespace {

visualization_msgs::Marker baseMarker(const std::string& frame_id,
                                      const ros::Time& stamp,
                                      const std::string& ns, int id,
                                      int32_t type) {
  visualization_msgs::Marker m;
  m.header.frame_id = frame_id;
  m.header.stamp = stamp;
  m.ns = ns;
  m.id = id;
  m.type = type;
  m.action = visualization_msgs::Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.color.a = 1.0;
  return m;
}

}  // namespace

visualization_msgs::MarkerArray buildRecognizedObstaclesMarkers(
    const std::vector<LockedStatic>& locked,
    const std::vector<Cluster>& clusters,
    const std::string& frame_id, const ros::Time& stamp,
    double sphere_scale_m, double lifetime_s) {
  visualization_msgs::MarkerArray arr;
  // Clean previous markers up to a sane cap so RViz doesn't accumulate stale
  // ones across ticks.
  visualization_msgs::Marker clear;
  clear.header.frame_id = frame_id;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::Marker::DELETEALL;
  arr.markers.push_back(clear);

  int id = 0;
  for (const auto& cl : clusters) {
    auto m = baseMarker(frame_id, stamp, "clusters", id++,
                        visualization_msgs::Marker::SPHERE);
    m.pose.position.x = cl.centroid.x;
    m.pose.position.y = cl.centroid.y;
    m.pose.position.z = 0.15;
    m.scale.x = m.scale.y = m.scale.z = sphere_scale_m;
    m.color.r = 0.2;
    m.color.g = 0.6;
    m.color.b = 1.0;
    if (lifetime_s > 0.0) m.lifetime = ros::Duration(lifetime_s);
    arr.markers.push_back(m);
  }
  for (const auto& ls : locked) {
    if (!ls.locked) continue;
    auto m = baseMarker(frame_id, stamp, "locked_static", id++,
                        visualization_msgs::Marker::SPHERE);
    m.pose.position.x = ls.centroid.x;
    m.pose.position.y = ls.centroid.y;
    m.pose.position.z = 0.30;
    m.scale.x = m.scale.y = m.scale.z = sphere_scale_m * 1.6;
    m.color.r = 1.0;
    m.color.g = 0.6;
    m.color.b = 0.0;
    if (lifetime_s > 0.0) m.lifetime = ros::Duration(lifetime_s);
    arr.markers.push_back(m);
  }
  return arr;
}

visualization_msgs::MarkerArray buildBlockingObstacleMarkers(
    WorldXY blocker, bool active, const std::string& frame_id,
    const ros::Time& stamp, double radius_m, double lifetime_s) {
  visualization_msgs::MarkerArray arr;
  visualization_msgs::Marker clear;
  clear.header.frame_id = frame_id;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::Marker::DELETEALL;
  arr.markers.push_back(clear);
  if (!active) return arr;

  auto cyl = baseMarker(frame_id, stamp, "blocker", 0,
                        visualization_msgs::Marker::CYLINDER);
  cyl.pose.position.x = blocker.x;
  cyl.pose.position.y = blocker.y;
  cyl.pose.position.z = 0.40;
  cyl.scale.x = cyl.scale.y = radius_m * 2.0;
  cyl.scale.z = 0.80;
  cyl.color.r = 1.0;
  cyl.color.g = 0.1;
  cyl.color.b = 0.1;
  cyl.color.a = 0.45;
  if (lifetime_s > 0.0) cyl.lifetime = ros::Duration(lifetime_s);
  arr.markers.push_back(cyl);

  auto label = baseMarker(frame_id, stamp, "blocker", 1,
                          visualization_msgs::Marker::TEXT_VIEW_FACING);
  label.pose.position.x = blocker.x;
  label.pose.position.y = blocker.y;
  label.pose.position.z = 1.10;
  label.scale.z = 0.35;
  label.color.r = label.color.g = label.color.b = 1.0;
  label.text = "BLOCKER";
  if (lifetime_s > 0.0) label.lifetime = ros::Duration(lifetime_s);
  arr.markers.push_back(label);
  return arr;
}

visualization_msgs::MarkerArray buildPathHistoryMarkers(
    const std::vector<WorldXY>& world_path, const std::string& frame_id,
    const ros::Time& stamp) {
  visualization_msgs::MarkerArray arr;
  visualization_msgs::Marker clear;
  clear.header.frame_id = frame_id;
  clear.header.stamp = stamp;
  clear.action = visualization_msgs::Marker::DELETEALL;
  arr.markers.push_back(clear);
  if (world_path.size() < 2) return arr;
  auto strip = baseMarker(frame_id, stamp, "history", 0,
                          visualization_msgs::Marker::LINE_STRIP);
  strip.scale.x = 0.06;
  strip.color.r = 0.2;
  strip.color.g = 1.0;
  strip.color.b = 0.3;
  strip.color.a = 0.9;
  for (const auto& p : world_path) {
    geometry_msgs::Point pt;
    pt.x = p.x;
    pt.y = p.y;
    pt.z = 0.05;
    strip.points.push_back(pt);
  }
  arr.markers.push_back(strip);
  return arr;
}

nav_msgs::OccupancyGrid buildOverlayGrid(const OccupancyView& g,
                                         const std::vector<uint8_t>& blocked,
                                         const std::string& frame_id,
                                         const ros::Time& stamp) {
  nav_msgs::OccupancyGrid og;
  og.header.frame_id = frame_id;
  og.header.stamp = stamp;
  og.info.resolution = static_cast<float>(g.resolution_m);
  og.info.width = static_cast<uint32_t>(g.width);
  og.info.height = static_cast<uint32_t>(g.height);
  og.info.origin.position.x = g.origin.x;
  og.info.origin.position.y = g.origin.y;
  og.info.origin.orientation.w = 1.0;
  og.data.resize(blocked.size());
  for (std::size_t i = 0; i < blocked.size(); ++i) {
    og.data[i] = blocked[i] ? 100 : 0;
  }
  return og;
}

std_msgs::String buildDebugText(PathMode mode, bool nominal_blocked,
                                bool candidate, bool cached_drivable,
                                double endpoint_distance_m,
                                std::size_t clusters,
                                std::size_t obstacle_pts) {
  std_msgs::String s;
  std::ostringstream os;
  os << "cpp_planner mode=" << pathModeToString(mode)
     << " blocked=" << (nominal_blocked ? "yes" : "no")
     << " candidate=" << (candidate ? "yes" : "no")
     << " cached_drivable=" << (cached_drivable ? "yes" : "no")
     << " endpoint=" << endpoint_distance_m
     << " clusters=" << clusters
     << " obs_pts=" << obstacle_pts;
  s.data = os.str();
  return s;
}

}  // namespace clr
