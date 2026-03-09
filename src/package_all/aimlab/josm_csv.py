#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import xml.etree.ElementTree as ET


def read_latlon_csv(csv_file_path):
    pts = []
    with open(csv_file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = float(row["lat"])
            lon = float(row["lon"])
            pts.append((lat, lon))
    return pts


def write_osm(points, osm_file_path):
    root = ET.Element("osm", attrib={"version": "0.6", "generator": "josm_csv.py"})
    node_ids = []
    for i, (lat, lon) in enumerate(points):
        nid = str(-(i + 1))
        node_ids.append(nid)
        ET.SubElement(
            root,
            "node",
            attrib={
                "id": nid,
                "action": "modify",
                "visible": "true",
                "lat": "{:.8f}".format(lat),
                "lon": "{:.8f}".format(lon),
            },
        )

    way = ET.SubElement(root, "way", attrib={"id": "-1", "action": "modify", "visible": "true"})
    for nid in node_ids:
        ET.SubElement(way, "nd", attrib={"ref": nid})
    ET.SubElement(way, "tag", attrib={"k": "highway", "v": "service"})
    ET.SubElement(way, "tag", attrib={"k": "name", "v": "csv_generated"})

    out_dir = os.path.dirname(osm_file_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = osm_file_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"<?xml version='1.0' encoding='UTF-8'?>\n")
        f.write(ET.tostring(root, encoding="utf-8"))
        f.write(b"\n")
    os.replace(tmp, osm_file_path)


def convert_csv_to_osm(csv_file_path, osm_file_path):
    points = read_latlon_csv(csv_file_path)
    if len(points) < 2:
        raise RuntimeError("CSV has fewer than 2 points")
    write_osm(points, osm_file_path)


def main():
    ap = argparse.ArgumentParser(description="Convert lat/lon CSV to single-way OSM")
    ap.add_argument("--csv", required=True, help="Input CSV path (header: lat,lon)")
    ap.add_argument("--osm", required=True, help="Output OSM path")
    args = ap.parse_args()
    convert_csv_to_osm(args.csv, args.osm)
    print(f"[ok] OSM saved: {args.osm}")


if __name__ == "__main__":
    main()
