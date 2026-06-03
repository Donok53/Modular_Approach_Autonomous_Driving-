#pragma once

#include <string>
#include <vector>

#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "constrained_local_replanner_cpp/avoidance_state.hpp"
#include "constrained_local_replanner_cpp/types.hpp"

namespace clr {

// Build a MarkerArray that draws every confirmed (locked) static obstacle as
// a coloured sphere. Mirrors the Python "recognized_obstacles" topic shape.
visualization_msgs::MarkerArray buildRecognizedObstaclesMarkers(
    const std::vector<LockedStatic>& locked,
    const std::vector<Cluster>& clusters,
    const std::string& frame_id, const ros::Time& stamp,
    double sphere_scale_m, double lifetime_s);

// Build a MarkerArray that highlights the blocker that triggered avoidance.
// One CYLINDER + one TEXT label.
visualization_msgs::MarkerArray buildBlockingObstacleMarkers(
    WorldXY blocker, bool active, const std::string& frame_id,
    const ros::Time& stamp, double radius_m, double lifetime_s);

// Visualise the candidate / cached local path as a MarkerArray strip so it
// shows up under the same RViz config the Python node used for
// path_history.
visualization_msgs::MarkerArray buildPathHistoryMarkers(
    const std::vector<WorldXY>& world_path, const std::string& frame_id,
    const ros::Time& stamp);

// Republish the blocked mask as an OccupancyGrid so the original RViz
// overlay panel keeps working. The mask is the same one we feed the
// avoidance planner each tick.
nav_msgs::OccupancyGrid buildOverlayGrid(const OccupancyView& g,
                                         const std::vector<uint8_t>& blocked,
                                         const std::string& frame_id,
                                         const ros::Time& stamp);

// A single TextViewFacing marker reporting the planner's current state for
// quick eyeballing. Mirrors the Python debug_text channel.
std_msgs::String buildDebugText(PathMode mode, bool nominal_blocked,
                                bool candidate, bool cached_drivable,
                                double endpoint_distance_m,
                                std::size_t clusters, std::size_t obstacle_pts);

}  // namespace clr
