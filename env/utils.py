from math import *
from typing import Tuple, List

R: float = 6371393.0
DEG_to_RAD: float = pi / 180.0
RAD_to_DEG: float = 180.0 / pi
KM2M: float = 1000.0
M2KM: float = 0.001
KT2MPS: float = 0.514444444444444
NM2M: float = 1852


def distance(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    """
    两个坐标点的距离
    """
    lng0: float = p0[0] * DEG_to_RAD
    lat0: float = p0[1] * DEG_to_RAD
    lng1: float = p1[0] * DEG_to_RAD
    lat1: float = p1[1] * DEG_to_RAD
    tmp1: float = sin((lat0 - lat1) / 2)
    tmp2: float = sin((lng0 - lng1) / 2)
    a: float = tmp1 * tmp1 + cos(lat0) * cos(lat1) * tmp2 * tmp2
    c: float = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def bearing(p0: Tuple[float, float], p1: Tuple[float, float]) -> float:
    """
    计算两个点的方向（p0 --> p1）
    """
    lng0: float = p0[0] * DEG_to_RAD
    lat0: float = p0[1] * DEG_to_RAD
    lng1: float = p1[0] * DEG_to_RAD
    lat1: float = p1[1] * DEG_to_RAD
    d_lng: float = lng1 - lng0
    tmp1: float = sin(d_lng) * cos(lat1)
    tmp2: float = cos(lat0) * sin(lat1) - sin(lat0) * cos(lat1) * cos(d_lng)
    return (atan2(tmp1, tmp2) * RAD_to_DEG) % 360


def move(src: Tuple[float, float], course: float, d: float) -> Tuple[float, float]:
    """
    从点src朝着course方向移动d，到达下一个点
    """
    lng1: float = radians(src[0])
    lat1: float = radians(src[1])
    course: float = radians(course)
    r: float = d / R
    lat2: float = asin(sin(lat1) * cos(r) + cos(lat1) * sin(r) * cos(course))
    lng2: float = lng1 + atan2(sin(course) * sin(r) * cos(lat1), cos(r) - sin(lat1) * sin(lat2))
    return degrees(lng2), degrees(lat2)


def calc_turn_prediction(spd: float, target: float, src: float, turn_rate: float) -> float:
    """
    计算转弯提前量
    """
    turn_angle: float = (target - src) % 360
    if turn_angle > 180:
        turn_angle -= 360
    return spd / radians(turn_rate) * tan(radians(abs(turn_angle) / 2))


def border_float(x: float, min_v: float, max_v: float) -> float:
    """
    当x小于最小值时，取最小值；当x大于最大值时，取最大值（x为float）
    """
    return min(max(x, min_v), max_v)


def border_int(x: int, min_v: int, max_v: int) -> int:
    """
    当x小于最小值时，取最小值；当x大于最大值时，取最大值（x为int）
    """
    return min(max(x, min_v), max_v)


def area(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    已知三个点的坐标（a,b,c），计算其三角形面积
    """
    ab: float = distance(a, b)
    ac: float = distance(a, c)
    bc: float = distance(b, c)
    p: float = (ab + ac + bc) / 2
    return sqrt(abs(p * (p - ab) * (p - ac) * (p - bc)))


def high(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """
    已知三个点的坐标（a,b,c），计算a点到bc的距离
    """
    return 2 * area(a, b, c) / distance(b, c)


def pnpoly(vertices: Tuple[Tuple[float, float]], p: Tuple[float, float]) -> bool:
    """
    判断一个2维点（p）是否在一个不规则多边形（vertices）内，3维是否可用尚未验证
    """
    n: int = len(vertices)
    j: int = n - 1
    in_poly: bool = False

    for i in range(n):
        if (vertices[i][1] > p[1]) != (vertices[j][1] > p[1]) and \
                p[0] < (vertices[j][0] - vertices[i][0]) * (p[1] - vertices[i][1]) / (
                vertices[j][1] - vertices[i][1]) + vertices[i][0]:
            in_poly = not in_poly
        j = i
    return in_poly


def make_bbox_3d(p: Tuple[float, float, float], ext: Tuple[float, float, float]) \
        -> Tuple[float, float, float, float, float, float]:
    """
    以3维点p为中心的长方体 -> (min_x, min_y, min_z, max_x, max_y, max_z)
    """
    return (p[0] - ext[0], p[1] - ext[1], p[2] - ext[2],
            p[0] + ext[0], p[1] + ext[1], p[2] + ext[2])


def make_bbox_2d(p: Tuple[float, float], ext: Tuple[float, float]) -> Tuple[float, float, float, float]:
    """
    以2维点p为中心的长方形 -> (min_x, min_y, max_x, max_y)
    """
    return (p[0] - ext[0], p[1] - ext[1],
            p[0] + ext[0], p[1] + ext[1])


def bbox_points_2d(p: Tuple[float, float], ext: Tuple[float, float]):
    """
    2维bbox的四个顶点坐标（如果闭合，则是五个）
    """
    min_lng, min_lat, max_lng, max_lat = make_bbox_2d(p, ext)
    return (
        (min_lng, min_lat),
        (min_lng, max_lat),
        (max_lng, max_lat),
        (max_lng, min_lat),
        (min_lng, min_lat)
    )


def position_in_bbox(bbox: Tuple[float, float, float, float, float, float], p: Tuple[float, float, float],
                     unit: Tuple[float]) -> int:
    """
    将空域网格化后，已知3维点p的位置和空域的经纬度范围bbox，求p在哪个网格里
    """
    x_size = int((bbox[3] - bbox[0]) / unit[0])
    y_size = int((bbox[4] - bbox[1]) / unit[1])
    z_size = int((bbox[5] - bbox[2]) / unit[2])
    x = int((p[0] - bbox[0]) / unit[0])
    y = int((p[1] - bbox[1]) / unit[1])
    z = int((p[2] - bbox[2]) / unit[2])
    if x >= x_size or y >= y_size or z >= z_size:
        return -1
    return x + y * x_size + z * x_size * y_size


def mid_point(p0: Tuple[float], p1: Tuple[float]):
    return ((x1 - x2) / 2 + x1 for x1, x2 in zip(p0, p1))


def get_split_lines(num_block: int, border: Tuple[float, float, float, float]):
    """
    # 将空域分为n等块
    """
    num: int = int(sqrt(num_block))
    x, y = (border[1] - border[0]) / num, (border[3] - border[2]) / num
    coord_list = []
    for i in range(num + 1):
        coord_list.append([(border[0], border[2] + i * y),
                           (border[1], border[2] + i * y)])
        coord_list.append([(border[0] + i * x, border[2]),
                           (border[0] + i * x, border[3])])
    return coord_list


def in_which_block(p: Tuple[float, float], border: Tuple[float, float, float, float], num_block: int) -> int:
    """
    2维点p在哪一个block里（一共有num_block个block）
    """
    num: int = int(sqrt(num_block))
    x, y = (border[1] - border[0]) / num, (border[3] - border[2]) / num
    i: int = border_int(int((p[0] - border[0]) / x), min_v=0, max_v=num - 1)
    j: int = border_int(int((p[1] - border[2]) / y), min_v=0, max_v=num - 1)
    return int(i * num + j)


def intersection(a0: float, a1: float) -> float:
    """
    两个角度的夹角[-180,180]
    """
    diff: float = (a0 - a1) % 360
    if diff > 180:
        return diff - 360
    return diff


def make_closest_bbox(points, ext):
    """
    给定一些3维位置点，画这些点组成的外接bbox
    """
    lng_lst, lat_lst, alt_lst = [], [], []
    for p in points:
        lng_lst.append(p[0])
        lat_lst.append(p[1])
        alt_lst.append(p[2])

    return (min(lng_lst) - ext[0], min(lat_lst) - ext[1], min(alt_lst) - ext[2],
            max(lng_lst) + ext[0], max(lat_lst) + ext[1], max(alt_lst) + ext[2])


def equal(list_a, list_b):
    """
    两个list是否完全相等（元素相等，忽略顺序）
    """
    if len(list_a) != len(list_b):
        return False
    return sum([a not in list_b for a in list_a]) <= 0


def calc_level(alt: float, v_spd: float, delta: float) -> float:
    """
    # 修正高度到高度层, 8300 → 8400
    """
    flight_level: List[float] = [i * 300.0 + 200.0 * int(i >= 29) for i in range(0, 50)]  # 0~8400,8900~12200

    num_lvl: int = int(delta / 300.0)

    lvl: float = int(alt / 300.0) * 300.0
    if alt < 8700.0:
        idx = flight_level.index(lvl)
        if (v_spd > 0 and alt - lvl != 0) or (v_spd == 0 and alt - lvl > 150):
            idx += 1
        return flight_level[idx + num_lvl]
    lvl += 200.0

    idx: int = flight_level.index(lvl)
    if v_spd > 0 and alt - lvl > 0:
        idx += 1
    elif v_spd < 0 and alt - lvl < 0:
        idx -= 1
    return flight_level[idx + num_lvl]
