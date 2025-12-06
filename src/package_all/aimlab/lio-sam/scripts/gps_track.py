import rosbag
import xml.etree.ElementTree as ET

bag = rosbag.Bag("2025-05-08-16-33-39.bag")
gpx = ET.Element('gpx', version="1.1", creator="rosbag2gpx")
trk = ET.SubElement(gpx, 'trk')
trkseg = ET.SubElement(trk, 'trkseg')

for topic, msg, t in bag.read_messages(topics=['/smc_2000/fix']):
    if msg.status.status >= 0:
        trkpt = ET.SubElement(trkseg, 'trkpt', lat=str(msg.latitude), lon=str(msg.longitude))
        ET.SubElement(trkpt, 'ele').text = str(msg.altitude)

tree = ET.ElementTree(gpx)
tree.write("gps_track.gpx", encoding='utf-8', xml_declaration=True)
print("GPX saved to gps_track.gpx")
