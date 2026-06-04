#include <thread>

#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <rviz/display_context.h>
#include <rviz/tool.h>
#include <rviz/tool_manager.h>
#include <std_srvs/Trigger.h>

#include <QColor>
#include <QFont>
#include <QIcon>
#include <QPainter>
#include <QPixmap>
#include <QString>

namespace lio_sam_rviz_plugins
{

class MapExtensionTriggerTool : public rviz::Tool
{
public:
  MapExtensionTriggerTool(const QString& name,
                          const QString& description,
                          const std::string& service_name,
                          const QColor& color,
                          const QString& icon_text)
    : name_(name)
    , description_(description)
    , service_name_(service_name)
    , color_(color)
    , icon_text_(icon_text)
  {
    shortcut_key_ = 0;
  }

  void onInitialize() override
  {
    setName(name_);
    setDescription(description_);
    setIcon(makeIcon(color_, icon_text_));
  }

  void activate() override
  {
    setStatus(QString("%1 requested").arg(name_));
    callServiceAsync(service_name_, name_.toStdString());
    returnToDefaultTool();
  }

  void deactivate() override
  {
  }

private:
  static QIcon makeIcon(const QColor& color, const QString& text)
  {
    QPixmap pixmap(48, 48);
    pixmap.fill(Qt::transparent);

    QPainter painter(&pixmap);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setBrush(color);
    painter.setPen(Qt::NoPen);
    painter.drawRoundedRect(3, 3, 42, 42, 6, 6);

    QFont font = painter.font();
    font.setBold(true);
    font.setPointSize(text.size() > 4 ? 7 : 9);
    painter.setFont(font);
    painter.setPen(Qt::white);
    painter.drawText(pixmap.rect(), Qt::AlignCenter, text);
    return QIcon(pixmap);
  }

  static void callServiceAsync(const std::string& service_name, const std::string& tool_name)
  {
    std::thread([service_name, tool_name]() {
      ros::NodeHandle nh;
      ros::ServiceClient client = nh.serviceClient<std_srvs::Trigger>(service_name);
      if (!client.waitForExistence(ros::Duration(0.8)))
      {
        ROS_WARN("map extension RViz tool %s: service unavailable: %s",
                 tool_name.c_str(),
                 service_name.c_str());
        return;
      }

      std_srvs::Trigger srv;
      if (!client.call(srv))
      {
        ROS_WARN("map extension RViz tool %s: service call failed: %s",
                 tool_name.c_str(),
                 service_name.c_str());
        return;
      }

      if (srv.response.success)
      {
        ROS_INFO("map extension RViz tool %s: %s", tool_name.c_str(), srv.response.message.c_str());
      }
      else
      {
        ROS_WARN("map extension RViz tool %s: %s", tool_name.c_str(), srv.response.message.c_str());
      }
    }).detach();
  }

  void returnToDefaultTool()
  {
    if (context_ == nullptr)
    {
      return;
    }
    rviz::ToolManager* tool_manager = context_->getToolManager();
    if (tool_manager == nullptr)
    {
      return;
    }
    rviz::Tool* default_tool = tool_manager->getDefaultTool();
    if (default_tool != nullptr && default_tool != this)
    {
      tool_manager->setCurrentTool(default_tool);
    }
  }

  QString name_;
  QString description_;
  std::string service_name_;
  QColor color_;
  QString icon_text_;
};

class MapExtensionStartTool : public MapExtensionTriggerTool
{
public:
  MapExtensionStartTool()
    : MapExtensionTriggerTool("START EXT",
                              "Start RViz-controlled map extension after localization is ready.",
                              "/map_extension/start",
                              QColor(20, 160, 80),
                              "START")
  {
  }
};

class MapExtensionSaveTool : public MapExtensionTriggerTool
{
public:
  MapExtensionSaveTool()
    : MapExtensionTriggerTool("SAVE EXT",
                              "Save and merge the active map extension.",
                              "/map_extension/finish",
                              QColor(30, 95, 210),
                              "SAVE")
  {
  }
};

class MapExtensionCancelTool : public MapExtensionTriggerTool
{
public:
  MapExtensionCancelTool()
    : MapExtensionTriggerTool("CANCEL EXT",
                              "Cancel the active map extension without merging it.",
                              "/map_extension/cancel",
                              QColor(210, 45, 45),
                              "STOP")
  {
  }
};

}  // namespace lio_sam_rviz_plugins

PLUGINLIB_EXPORT_CLASS(lio_sam_rviz_plugins::MapExtensionStartTool, rviz::Tool)
PLUGINLIB_EXPORT_CLASS(lio_sam_rviz_plugins::MapExtensionSaveTool, rviz::Tool)
PLUGINLIB_EXPORT_CLASS(lio_sam_rviz_plugins::MapExtensionCancelTool, rviz::Tool)
