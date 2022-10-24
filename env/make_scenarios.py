import time

import numpy as np
import pymongo
import matplotlib.pyplot as plt

from env.core import AircraftAgentSet
from env.load import load_data_set
from env.model import Routing, FlightPlan, Aircraft
from env.render import plot_line, make_random_color, make_color
from env.utils import pnpoly

"""
 1. 找到所有经过武汉扇区（vertices）的航路 → wh_routing_list；
 2. 截取wh_routing_list航路中在武汉扇区（vertices）里的航段；
 3. 随机抽取120个routing，构建飞行计划和AgentSet；
 4. 运行AgentSet，并进行冲突探测；
 5. 剔除冲突时间-起飞时间<=600的飞行计划，并重建AgentSet；
 6. 运行AgentSet，并进行冲突探测；
 7. 记录冲突信息和飞行计划 → meta_scenarios；
 8. 记录各个冲突信息和飞行计划 → scenarios_gail；
"""

shift = 360

vertices = ((109.51666666666667, 31.9),
            (110.86666666666666, 33.53333333333333),
            (114.07, 32.125),
            (115.81333333333333, 32.90833333333333),
            (115.93333333333334, 30.083333333333332),
            (114.56666666666666, 29.033333333333335),
            (113.12, 29.383333333333333),
            (109.4, 29.516666666666666),
            (109.51666666666667, 31.9),
            (109.51666666666667, 31.9))

data_set = load_data_set()
flight_level = {i-20: i * 300.0 + int(i >= 29) * 200.0 for i in range(20, 41)}  # 6000~12500


# step 1和2
def search_routing_in_wuhan():
    import simplekml

    kml = simplekml.Kml()
    folder = kml.newfolder(name='Polygons')
    plot_line(folder,
              (p+(8100.0, ) for p in vertices),
              name='border')

    folder = kml.newfolder(name='Lines')
    inner_routes, route_id = [], []
    for r_id, routing in data_set.rou_dict.items():
        wpt_list = routing.wpt_list

        plot_line(folder,
                  (wpt.location.tuple()+(8100.0, ) for wpt in wpt_list),
                  name=r_id,
                  color=make_random_color())

        i, idx = 0, []
        for i, wpt in enumerate(wpt_list):
            if pnpoly(vertices, wpt.location.tuple()):
                idx.append(i)

        if len(idx) > 0:
            min_v, max_v = min(idx), max(idx)
            if min_v == 0:
                mode = 'start'
            elif max_v == i:
                mode = 'end'
            else:
                mode = 'part'

            wpt_list_part = wpt_list[max(0, min_v-1): min(i+1, max_v+2)]
            inner_routes.append([r_id, wpt_list_part, mode])
            route_id.append(r_id)

            plot_line(folder,
                      (wpt.location.tuple() + (8100.0,) for wpt in wpt_list_part),
                      name=r_id+'_part',
                      color=make_color(255, 0, 0))
    kml.save('wuhan.kml')

    count = 0
    for fpl in data_set.fpl_dict.values():
        if fpl.routing.id in route_id:
            count += 1
    print(count)

    return inner_routes


# step 3
def get_fpl_random(routes, interval=30, number=100, max_time=43200):
    fpl_set = list(data_set.fpl_dict.values())
    np.random.shuffle(fpl_set)

    count = 0
    fpl_list, candi = [], {}
    for j in range(0, max_time, interval):
        np.random.shuffle(routes)
        candi[j] = []

        for [r_id, wpt_list, mode] in routes[:number]:
            fpl_template = fpl_set[count % len(fpl_set)]

            # routing
            routing = Routing(id=r_id, wpt_list=wpt_list)
            # aircraft
            aircraft = fpl_template.aircraft
            # min_alt
            if mode == 'start':  # 在扇区内起飞的航班，上升后平飞
                min_alt = 6000.0
            else:
                min_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            # max_alt
            if mode == 'end':   # 在扇区内落地的航班，下降
                max_alt = 6000.0
            elif np.random.randint(0, 60) % 2 == 0:  # 1/2的航班起始高度等于目标高度
                max_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            else:
                max_alt = min_alt
            # flight plan
            fpl = FlightPlan(id=str(count),
                             routing=routing,
                             aircraft=aircraft,
                             startTime=j,
                             min_alt=min_alt,
                             max_alt=max_alt)
            fpl_list.append(fpl)
            candi[j].append(fpl.id)
            count += 1

    return fpl_list, candi


# step 4
def run_scenario(fpl_list, candi):
    print('\t>>>', len(fpl_list))
    agent_set = AircraftAgentSet(fpl_list=fpl_list, candi=candi)
    agent_set.step(1)

    all_conflicts, record = [], {'flow': [], 'conflict': []}
    shift_list = []

    step_sum = 0.0
    detect_sum = 0.0
    build_sum = 0.0
    compair_sum = 0.0
    compute_sum = 0.0
    while True:
        start = time.time()
        _, build, compair, compute = agent_set.step(duration=8)
        step_sum += time.time()-start

        start = time.time()
        conflicts, _, _ = agent_set.detect()
        detect_sum += time.time() - start
        build_sum += build
        compair_sum += compair
        compute_sum += compute

        for c in conflicts:
            [a0, a1] = c.id.split('-')
            fpl0 = agent_set.agents[a0].fpl
            fpl1 = agent_set.agents[a1].fpl
            if c.time - fpl0.startTime < shift:
                shift_list.append(a0)
            if c.time - fpl1.startTime < shift:
                shift_list.append(a1)

        now = agent_set.time
        if len(conflicts) > 0:
            all_conflicts += conflicts

        if now % 1000 == 0:
            # print('\t>>>', now, len(agent_set.agent_id_en), len(all_conflicts))
            # print('\t\t>>>', step_sum, detect_sum, build_sum, compair_sum, compute_sum)
            print('{},{},{},{},{}'.format(step_sum, detect_sum, build_sum, compair_sum, compute_sum))
            step_sum = 0.0
            detect_sum = 0.0
            build_sum = 0.0
            compair_sum = 0.0
            compute_sum = 0.0

        record['flow'].append([now, len(agent_set.agent_id_en)])
        record['conflict'].append([now, len(conflicts)])

        if agent_set.is_done():
            print('场景运行结束：', now, len(conflicts), time.time() - start)
            break

    return all_conflicts, record, shift_list


# Step 7
def write_in_db(name, conflict_info, fpl_info):
    database = pymongo.MongoClient('localhost')['admin']
    collection = database['scenarios_new']

    print(len(conflict_info))
    c_times = [c.time for c in conflict_info]
    conflicts = [c.to_dict() for c in conflict_info]
    fpl_list = [fpl.to_dict() for fpl in fpl_info]
    collection.insert_one(dict(id=name, start=min(c_times)-301, conflicts=conflicts, fpl_list=fpl_list))


def analysis(record, new_record, candi, new_candi):
    fig, axes = plt.subplots(3, 1)
    x, y = [], []
    for [t, flow, *_] in record['flow']:
        x.append(t)
        y.append(flow)
    axes[0].plot(x, y, label='before')
    x, y = [], []
    for [t, flow, *_] in new_record['flow']:
        x.append(t)
        y.append(flow)
    axes[0].plot(x, y, label='after')
    axes[0].set_xlabel('Time Axis')
    axes[0].set_ylabel('Flow')
    axes[0].legend()

    x, y = [], []
    for [t, flow, *_] in record['conflict']:
        x.append(t)
        y.append(flow)
    axes[1].plot(x, y, label='before')
    x, y = [], []
    for [t, flow, *_] in new_record['conflict']:
        x.append(t)
        y.append(flow)
    axes[1].plot(x, y, label='after')
    axes[1].set_xlabel('Time Axis')
    axes[1].set_ylabel('Conflict')
    axes[1].legend()

    x, y = [], []
    for key in sorted(candi.keys()):
        x.append(key)
        y.append(len(candi[key]))
    axes[2].plot(x, y, label='before')
    x, y = [], []
    for key in sorted(new_candi.keys()):
        x.append(key)
        y.append(len(new_candi[key]))
    axes[2].plot(x, y, label='after')
    axes[2].set_xlabel('Time Axis')
    axes[2].set_ylabel('Candi')
    axes[2].legend()

    plt.subplots_adjust(hspace=0.5)
    fig.savefig('stats.pdf')
    plt.show()


def main(num=10):
    np.random.seed(1234)
    inner_routes = search_routing_in_wuhan()
    print('>>> 一共找到{}条经过武汉扇区的Routing（Step 1和2）'.format(len(inner_routes)))

    for i in range(0, num):
        print('>>> 随机加载航空器（Step 3)')
        fpl_list, candi = get_fpl_random(inner_routes[:], interval=20, number=10, max_time=40000)

        print('>>> 开始运行场景，并进行冲突探测（Step 4和5）')
        conflicts, record, shift_list = run_scenario(fpl_list, candi)

        new_fpl_list, new_candi = [], {}
        for fpl in fpl_list:
            if fpl.id not in shift_list:
                new_fpl_list.append(fpl)
                start = fpl.startTime
                if start in new_candi.keys():
                    new_candi[start].append(fpl.id)
                else:
                    new_candi[start] = [fpl.id]

        print('>>> 重新运行场景，并进行冲突探测（Step 6和5）')
        new_conflicts, new_record, shift_list = run_scenario(new_fpl_list, new_candi)
        assert len(shift_list) == 0
        analysis(record, new_record, candi, new_candi)

        print('>>> 记录冲突信息和飞行计划（Step 7）')
        write_in_db(i, new_conflicts, new_fpl_list)
        break


if __name__ == '__main__':
    main()
