import random
import copy

import cv2
import simplekml
import numpy as np

from .utils import move, NM2M, bbox_points_2d, pnpoly, get_split_lines

vertices_ = [
    (109.51666666666667, 31.9),
    (110.86666666666666, 33.53333333333333),
    (114.07, 32.125),
    (115.81333333333333, 32.90833333333333),
    (115.93333333333334, 30.083333333333332),
    (114.56666666666666, 29.033333333333335),
    (113.12, 29.383333333333333),
    (109.4, 29.516666666666666),
    (109.51666666666667, 31.9),
    (109.51666666666667, 31.9)]  # 武汉空域边界点

border_sector = (109.3, 116.0, 29.0, 33.5)  # 扇区实际的经纬度范围(min_x, max_y, min_y, max_y)
border_render = (109.0, 120.0, 26.0, 34.0)  # 扇区可视化的经纬度范围
border_property = {'color': (100, 100, 100), 'thickness': 2}  # BGR
segment_property = {'color': (107, 55, 19), 'thickness': 1}  # BGR
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4

alt_mode = simplekml.AltitudeMode.absolute

scale_: int = 100  # BGR
channel_: int = 3
decimal: int = 1
radius: int = 5
fps: int = 8
interval: int = 30


# ---------
# functions
# ---------
def resolution(border, scale: int):
    """
    分辨率（可视化界面的长和宽）
    假设border为[1.0, 9.0, 1.0, 7.0]，scale为100，则分辨率为800x600
    """
    min_x, max_x, min_y, max_y, *_ = border
    return (
        int((max_x - min_x) * scale),
        int((max_y - min_y) * scale)
    )


def convert_coord_to_pixel(points, border, scale: int):
    """
    将点坐标（lng, lat）转化为像素点的位置（x, y）
    """
    min_x, max_x, min_y, max_y, *_ = border
    scale_x = (max_x - min_x) * scale
    scale_y = (max_y - min_y) * scale
    return [
        (
            int((x - min_x) / (max_x - min_x) * scale_x),
            int((max_y - y) / (max_y - min_y) * scale_y)
        )
        for [x, y, *_] in points
    ]


class CVRender:
    def __init__(self, video_path, image_path):
        """
        用于录制视频
        """
        self.width, self.length = resolution(border_render, scale_)
        self.base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读取扇区底图（只包含扇区边界和航段线）
        self.video = cv2.VideoWriter(video_path,
                                     cv2.VideoWriter_fourcc(*'MJPG'),
                                     fps,
                                     (self.width, self.length))

    def render(self, scene, mode='human', wait=1):
        if self.video is None:
            return

        # 底图
        image = copy.deepcopy(self.base_image)

        # 网格线
        image = add_lines_on_base_map(get_split_lines(scene.A, border_sector),
                                      image,
                                      color=(106, 106, 255),
                                      display=False)
        record = scene.record
        conflict_acs = scene.conflict_acs

        # 全局信息
        info = scene.info
        global_info = {'>>> Information': 'No.{}_{}'.format(info['id'], len(info['fpl_list']))}
        image = add_texts_on_base_map(global_info, image, (750, 30), color=(128, 0, 128))

        # 冲突信息
        conflict_info = {'>>> Conflict': str(record['result'])}
        image = add_texts_on_base_map(conflict_info, image, (750, 80), color=(128, 0, 128))
        conflict_info = {'Real': ''}
        for i, c in enumerate(record['r_conflicts']):
            if i >= 4:
                conflict_info['r_n'] = '...'
                break
            conflict_info['r_' + str(i + 1)] = str(c)
        conflict_info['Fake'] = ''
        for i, c in enumerate(record['f_conflicts']):
            if i >= 4:
                conflict_info['f_n'] = '...'
                break
            conflict_info['f_' + str(i + 1)] = str(c)
        image = add_texts_on_base_map(conflict_info, image, (750, 100), color=(0, 0, 0))

        # 指令信息
        cmd_info = {'>>> Command & status': ''}
        ac_cmd_dict = {}
        for ac, cmd_list in record['cmd_info'].items():
            ret = {}
            for cmd in cmd_list:
                ret.update(cmd.to_dict())
            ac_cmd_dict[ac] = ret
        image = add_texts_on_base_map(cmd_info, image, (750, 320), color=(128, 0, 128))

        points_dict = record['tracks']
        for t in sorted(points_dict.keys()):
            points = points_dict[t]
            frame = copy.deepcopy(image)

            # 运行信息，读秒、在航路上的飞机数量等
            texts = {'Time': '{}({}), ac_en: {}, speed: x{}'.format(t, scene.now(), len(points), 1000/wait)}
            frame = add_texts_on_base_map(texts, frame, (750, 50), color=(0, 0, 0))

            # 轨迹点
            frame = add_points_on_base_map(points, frame, conflict_ac=conflict_acs, ac_cmd_dict=ac_cmd_dict)

            # 图片渲染
            cv2.imshow(mode, frame)
            button = cv2.waitKey(wait)
            if button == 113:    # 按q键退出渲染
                return
            elif button == 112:  # 按p键加速渲染
                wait = int(wait*0.1)
            elif button == 111:  # 按o键减速渲染
                wait *= 10
            else:
                self.video.write(frame)

    def close(self):
        if self.video is not None:
            self.video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()
            self.video = None


# ---------
# opencv
# ---------
def add_points_on_base_map(points, image, color=(0, 0, 0), **kwargs):
    """
    在image里增加点（图标为圆）
    """
    count = 0
    for [name, lng, lat, alt, *point] in points:
        coord = (lng, lat)
        coord_idx = convert_coord_to_pixel([coord], border=border_render, scale=scale_)[0]

        # 如果飞机是参与冲突的
        if name in kwargs['conflict_ac']:
            # 每个飞机都是个圆，圆的颜色代表飞行高度，Green（低）-Yellow-Red（高），
            range_mixed = min(510, max((alt - 6000) / 4100 * 510, 0))
            if range_mixed <= 255:
                cv2.circle(image, coord_idx, radius, (0, 255, range_mixed), -1)
            else:
                cv2.circle(image, coord_idx, radius, (0, 510 - range_mixed, 255), -1)
            # 紫色直线代表其航向，长度代表速度
            heading_spd_point = move(coord, point[-1], 600 / 3600 * point[0] * NM2M)
            add_lines_on_base_map([[coord, heading_spd_point, False]],
                                  image,
                                  color=(255, 0, 0),
                                  display=False,
                                  thickness=2)
            # 画观察范围
            bbox_points = bbox_points_2d(coord, (1.0, 1.0))
            for i, p in enumerate(bbox_points[:-1]):
                add_lines_on_base_map([[p, bbox_points[i + 1], False]],
                                      image,
                                      display=False)
            # 加上呼号
            cv2.putText(image, name, (coord_idx[0], coord_idx[1] + 10), font, font_scale, color, 1)
            cmd_dict = kwargs['ac_cmd_dict'][name]
            x, y = 750, 340 + count * 80
            cv2.putText(image, name, (x, y), font, font_scale, color, 1)
            state = 'Altitude: {}({})'.format(round(alt, decimal), cmd_dict['Altitude'])
            cv2.putText(image, state, (x + 20, y + 20), font, font_scale, color, 1)
            state = 'Speed: {}({})({})'.format(round(point[0], decimal), round(point[1], decimal), cmd_dict['Speed'])
            cv2.putText(image, state, (x + 20, y + 40), font, font_scale, color, 1)
            state = 'Heading: {}({})'.format(round(point[2], decimal), cmd_dict['Heading'])
            cv2.putText(image, state, (x + 20, y + 60), font, font_scale, color, 1)
            count += 1
        else:
            cv2.circle(image, coord_idx, int(radius * 0.6), (87, 139, 46), -1)

    return image


def add_texts_on_base_map(texts, image, pos, color=(255, 255, 255), thickness=1):
    """
    在image里增加文字
    """
    x, y = pos
    i = 0
    for key, text in texts.items():
        if isinstance(text, str):
            string = "{}: {}".format(key, text)
            cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, thickness)
            i += 1
        else:
            for j, text_ in enumerate(text):
                string = "{}_{}: {}".format(key, j + 1, text_)
                cv2.putText(image, string, (x, y + i * 20), font, font_scale, color, thickness)
                i += 1
    return image


def add_lines_on_base_map(lines, image, color=(255, 0, 255), display=True, thickness=1):
    """
    在image里增加线
    """
    if len(lines) <= 0:
        return image

    for [pos0, pos1, *other] in lines:
        [start, end] = convert_coord_to_pixel([pos0, pos1], border=border_render, scale=scale_)
        cv2.line(image, start, end, color, thickness)

        if display:
            cv2.putText(image,
                        ' H_dist: {}, V_dist: {}'.format(round(other[0], decimal), round(other[1], decimal)),
                        (int((start[0] + end[0]) / 2) + 10, int((start[1] + end[1]) / 2) + 10),
                        font,
                        font_scale,
                        color,
                        1)
    return image


# ---------
# simplekml
# ---------


def make_color(red, green, blue):
    return simplekml.Color.rgb(red, green, blue, 100)


def make_random_color():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return simplekml.Color.rgb(r, g, b, 100)


def plot_line(kml, points, name='line', color=simplekml.Color.white):
    line = kml.newlinestring(name=name,
                             coords=[tuple(p) for p in points],
                             altitudemode=alt_mode,
                             extrude=1)
    line.style.polystyle.color = color
    line.style.linestyle.color = color
    line.style.linestyle.width = 1


def plot_point(kml, point, name='point', hdg=None, description=None):
    point = kml.newpoint(name=name,
                         coords=[point],
                         altitudemode=alt_mode,
                         description=description)
    point.style.labelstyle.scale = 0.25
    point.style.iconstyle.icon.href = 'plane.png'
    if hdg is not None:
        point.style.iconstyle.heading = (hdg + 270) % 360


def draw(tracks=None, plan=None, save_path='simplekml'):
    kml = simplekml.Kml()

    if tracks is not None:
        folder = kml.newfolder(name='real')
        for key, points in tracks.items():
            plot_line(folder, points, name=key, color=simplekml.Color.chocolate)

    if plan is not None:
        folder = kml.newfolder(name='plan')
        for key, points in plan.items():
            plot_line(folder, points, name=key, color=simplekml.Color.royalblue)

    print("Save to " + save_path + ".kml successfully!")
    kml.save(save_path + '.kml')


# ---------
# CV Demo
# ---------
def search_routing_in_sector(vertices):
    """
    将所有经过该扇区的航路筛选出来
    """
    segments = {}
    check_list = []
    for key, routing in load_data_set().rou_dict.items():
        wpt_list = routing.wpt_list

        in_poly_idx = [i for i, wpt in enumerate(wpt_list) if pnpoly(vertices, wpt.location.tuple())]
        if len(in_poly_idx) <= 0:
            continue

        min_idx = max(min(in_poly_idx) - 1, 0)
        max_idx = min(len(wpt_list), max(in_poly_idx) + 2)
        new_wpt_list = wpt_list[min_idx:max_idx]
        assert len(new_wpt_list) >= 2

        for i, wpt in enumerate(new_wpt_list[1:]):
            last_wpt = new_wpt_list[i]

            name_f, name_b = last_wpt.id + '-' + wpt.id, wpt.id + '-' + last_wpt.id
            if name_f not in check_list:
                segments[name_f] = [last_wpt.location.array(), wpt.location.array()]
                check_list += [name_b, name_f]
    return segments


def generate_wuhan_base_map(vertices, border, scale, save_path=None, channel=3):
    """
    画出武汉扇区的边界线和扇区内的航段线，并保存为图片的形式
    """
    # 计算分辨率
    width, length = resolution(border_render, scale)
    # 创建一个的白色画布，RGB(255,255,255)为白色
    image = np.ones((length, width, channel), np.uint8) * 255
    # 将空域边界画在画布上
    points = convert_coord_to_pixel(vertices, border=border, scale=scale_)
    cv2.polylines(image,
                  [np.array(points, np.int32).reshape((-1, 1, 2,))],
                  isClosed=True,
                  color=border_property['color'],
                  thickness=border_property['thickness'])
    # 将航路段画在画布上
    segments = search_routing_in_sector(vertices)
    for seg in segments.values():
        seg_idx = convert_coord_to_pixel(seg, border=border, scale=scale)
        cv2.line(image,
                 seg_idx[0], seg_idx[1],
                 segment_property['color'],
                 segment_property['thickness'])
    # 制作成图片
    if save_path is not None:
        cv2.imwrite(save_path, image)
    # 按q结束展示
    cv2.imshow("wuhan", image)
    if cv2.waitKey(0) == 113:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    from .load import load_data_set

    generate_wuhan_base_map(vertices=vertices_,
                            border=border_render,
                            scale=scale_,
                            save_path='data/wuhan_base_render.png')
