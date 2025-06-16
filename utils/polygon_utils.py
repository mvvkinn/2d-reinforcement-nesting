import ezdxf
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union
import numpy as np

def arc_to_linestring(arc, segments=360):
    """
    ARC 엔티티를 LineString으로 근사변환
    :param arc: dxf ARC Entity
    :param segments: 근사 분할값
    :return:
    """
    from math import radians, cos, sin
    start_angle = arc.dxf.start_angle
    end_angle = arc.dxf.end_angle
    if end_angle < start_angle:
        end_angle += 360
    angle_step = (end_angle - start_angle) / segments
    cx = arc.dxf.center.x
    cy = arc.dxf.center.y
    r = arc.dxf.radius
    points = [
        (
            cx + r * cos(radians(start_angle + angle_step * i)),
            cy + r * sin(radians(start_angle + angle_step * i)),
        )
        for i in range(segments + 1)
    ]
    return LineString(points)

# CIRCLE 엔티티를 주어진 분할 수로 선분 형태로 변환
def circle_to_linestring(center, radius, segments=360):
    angles = np.linspace(0, 2 * np.pi, segments)
    points = [(center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)) for a in angles]
    points.append(points[0])  # 닫기
    return LineString(points)

# DXF 도면에서 각종 선형 엔티티(LINE, POLYLINE, ARC 등)를 추출하고 shapely LineString 리스트로 반환
# ARC는 세분화하여 곡선을 직선 근사 처리하며, 중심+반지름이 유사한 ARC는 하나의 원으로 병합
def extract_lines_from_dxf(doc):
    msp = doc.modelspace()
    lines = []

    # DXF 엔티티별로 선분/폴리라인/아크/원 추출
    for e in msp:
        # 색상이 없는 경우에만 외곽선에 해당하므로 그 외의 색상은 건너뛴다
        # if e.dxf.color != 256:
        #     continue

        if e.dxftype() == 'LINE':
            start = (e.dxf.start.x, e.dxf.start.y)
            end = (e.dxf.end.x, e.dxf.end.y)
            lines.append(LineString([start, end]))

        elif e.dxftype() == 'LWPOLYLINE':
            points = [tuple(p[:2]) for p in e.get_points()]
            if e.closed:
                points.append(points[0])
            lines.append(LineString(points))

        elif e.dxftype() == 'POLYLINE':
            points = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            if e.is_closed:
                points.append(points[0])
            lines.append(LineString(points))

        elif e.dxftype() == 'CIRCLE':
            center = (e.dxf.center.x, e.dxf.center.y)
            radius = e.dxf.radius
            lines.append(circle_to_linestring(center, radius))

        elif e.dxftype() == 'ARC':
            lines.append(arc_to_linestring(e, segments=360))

    # ARC들을 중심 좌표와 반지름으로 그룹화하여 원형 병합 시도
    from collections import defaultdict
    arc_groups = defaultdict(list)

    for e in msp:
        if e.dxftype() == 'ARC':
            center = (round(e.dxf.center.x, 1), round(e.dxf.center.y, 1))
            radius = round(e.dxf.radius, 1)
            arc_groups[(center, radius)].append(e)

    for (center, radius), arcs in arc_groups.items():
        if len(arcs) >= 3:  # 원형에 가까운 경우로 판단
            lines.append(circle_to_linestring(center, radius, segments=360))

    return lines

# 추출된 선분들을 병합하고 polygonize하여 닫힌 다각형 생성
# 가장 큰 외곽 polygon에서 내부 polygon들을 차집합 처리하여 구멍을 반영한 최종 폴리곤 생성
def merge_lines_with_holes(lines, buffer=0.05):
    merged = unary_union(lines)

    from shapely.ops import polygonize_full

    if isinstance(merged, (LineString, MultiLineString)):
        polygons, dangles, cuts, invalids = polygonize_full(merged.buffer(buffer))
        polys = list(polygons.geoms)
    else:
        print("merge 실패: 닫힌도형이 아닐수있음")
        return

    if len(polys) == 0:
        print("폴리곤 생성 실패")
        return

    # 중심 좌표 계산
    cx = np.mean([p.centroid.x for p in polys])
    cy = np.mean([p.centroid.y for p in polys])
    center = np.array([cx, cy])

    # 기준 면적
    max_area = max(p.area for p in polys)
    area_threshold = max_area * 0.5

    # 필터링: 충분히 넓고 중심에서 멀리 떨어진 후보만 outer 후보로 간주
    candidates = [p for p in polys if p.area > area_threshold]
    if len(candidates) > 1:
        outer = max(candidates, key=lambda p: np.linalg.norm(np.array([p.centroid.x, p.centroid.y]) - center))
    else:
        outer = max(polys, key=lambda p: p.area)

    # outer 제외한 나머지를 차집합 처리
    difference = outer
    for p in polys:
        if p != outer and outer.contains(p.centroid):
            difference = difference.difference(p)

    return difference

def ezdxf_to_polygon(doc):
    try:
      lines = extract_lines_from_dxf(doc)
      return merge_lines_with_holes(lines)
    except Exception as e:
      print(e)
      return None

    # 중심 좌표 계산
    cx = np.mean([p.centroid.x for p in polys])
    cy = np.mean([p.centroid.y for p in polys])
    center = np.array([cx, cy])

    # 기준 면적
    max_area = max(p.area for p in polys)
    area_threshold = max_area * 0.5

    # 필터링: 충분히 넓고 중심에서 멀리 떨어진 후보만 outer 후보로 간주
    candidates = [p for p in polys if p.area > area_threshold]
    if len(candidates) > 1:
        outer = max(candidates, key=lambda p: np.linalg.norm(np.array([p.centroid.x, p.centroid.y]) - center))
    else:
        outer = max(polys, key=lambda p: p.area)

    # outer 제외한 나머지를 차집합 처리
    difference = outer
    for p in polys:
        if p != outer and outer.contains(p.centroid):
            difference = difference.difference(p)

    return difference

def ezdxf_to_polygon(doc):
    try:
      lines = extract_lines_from_dxf(doc)
      return merge_lines_with_holes(lines)
    except Exception as e:
      print(e)
      return None

    # 중심 좌표 계산
    cx = np.mean([p.centroid.x for p in polys])
    cy = np.mean([p.centroid.y for p in polys])
    center = np.array([cx, cy])

    # 기준 면적
    max_area = max(p.area for p in polys)
    area_threshold = max_area * 0.5

    # 필터링: 충분히 넓고 중심에서 멀리 떨어진 후보만 outer 후보로 간주
    candidates = [p for p in polys if p.area > area_threshold]
    if len(candidates) > 1:
        outer = max(candidates, key=lambda p: np.linalg.norm(np.array([p.centroid.x, p.centroid.y]) - center))
    else:
        outer = max(polys, key=lambda p: p.area)

    # outer 제외한 나머지를 차집합 처리
    difference = outer
    for p in polys:
        if p != outer and outer.contains(p.centroid):
            difference = difference.difference(p)

    return difference

def ezdxf_to_polygon(doc):
    try:
      lines = extract_lines_from_dxf(doc)
      return merge_lines_with_holes(lines)
    except Exception as e:
      print(e)
      return None

    # 중심 좌표 계산
    cx = np.mean([p.centroid.x for p in polys])
    cy = np.mean([p.centroid.y for p in polys])
    center = np.array([cx, cy])

    # 기준 면적
    max_area = max(p.area for p in polys)
    area_threshold = max_area * 0.5

    # 필터링: 충분히 넓고 중심에서 멀리 떨어진 후보만 outer 후보로 간주
    candidates = [p for p in polys if p.area > area_threshold]
    if len(candidates) > 1:
        outer = max(candidates, key=lambda p: np.linalg.norm(np.array([p.centroid.x, p.centroid.y]) - center))
    else:
        outer = max(polys, key=lambda p: p.area)

    # outer 제외한 나머지를 차집합 처리
    difference = outer
    for p in polys:
        if p != outer and outer.contains(p.centroid):
            difference = difference.difference(p)

    return difference

def ezdxf_to_polygon(doc):
    try:
      lines = extract_lines_from_dxf(doc)
      return merge_lines_with_holes(lines)
    except Exception as e:
      print(e)
      return None