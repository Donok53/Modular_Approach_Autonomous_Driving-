import csv

def convert_csv_to_osm(csv_file_path, osm_file_path):
    # CSV 파일에서 경로 데이터 읽기
    trajectory = []
    with open(csv_file_path, 'r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            lat = row['lat']
            lon = row['lon']
            trajectory.append((lat, lon))

    # OSM 파일로 변환하여 저장
    with open(osm_file_path, 'w') as file:
        file.write('<?xml version=\'1.0\' encoding=\'UTF-8\'?>\n')
        file.write('<osm version=\'0.6\' generator=\'Python script\'>\n')
        
        # 노드 작성
        for i, point in enumerate(trajectory):
            file.write(f"  <node id='-{i+1}' action='modify' visible='true' lat='{point[0]}' lon='{point[1]}' />\n")
        
        # 웨이 작성
        file.write('  <way id=\'-1\' action=\'modify\' visible=\'true\'>\n')
        for i in range(len(trajectory)):
            file.write(f"    <nd ref='-{i+1}' />\n")
        file.write('  </way>\n')
        
        file.write('</osm>\n')

# CSV 파일 경로와 OSM 파일 경로 설정
csv_file_path = '/Users/user/Downloads/trajectory_wgs84.csv'
osm_file_path = '/Users/user/Desktop/test_trajectory.osm'

# CSV 파일을 OSM 파일로 변환
convert_csv_to_osm(csv_file_path, osm_file_path)
