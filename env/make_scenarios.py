import time

import pymongo
import matplotlib.pyplot as plt

from flightEnv import AircraftAgentSet

from env.model import Routing, FlightPlan
from env.render import *
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

min_time, max_time = 0, 2000
shift = 600

vertices = [(109.51666666666667, 31.9), (110.86666666666666, 33.53333333333333),
            (114.07, 32.125), (115.81333333333333, 32.90833333333333),
            (115.93333333333334, 30.083333333333332), (114.56666666666666, 29.033333333333335),
            (113.12, 29.383333333333333), (109.4, 29.516666666666666),
            (109.51666666666667, 31.9), (109.51666666666667, 31.9)]

data_set = load_data_set()
rou_set = list(data_set.routings.values())
fpl_set = list(data_set.flightPlans.values())
flight_level = [i * 300.0 for i in range(21, 29)]  # 6300~8400
flight_level += [i * 300.0 + 200.0 for i in range(29, 41)]  # 8900~12200
flight_level = {i: level for i, level in enumerate(flight_level)}


# step 1和2
def search_routing_in_wuhan(visual=False):
    inner_routes, plot_points = [], {}
    for routing in rou_set:
        i, in_poly_idx = 0, []
        for i, wpt in enumerate(routing.waypointList):
            if pnpoly(vertices, wpt.location.toArray()):
                in_poly_idx.append(i)

        if len(in_poly_idx) > 0:
            min_v, max_v = min(in_poly_idx), max(in_poly_idx)

            if min_v == 0 and max_v == i:
                continue

            if min_v == 0:
                mode, section = 'start', list(range(0, max_v+2))
            elif max_v == i:
                mode, section = 'end', list(range(min_v-1, i+1))
            else:
                mode, section = 'part', list(range(min_v-1, max_v+2))

            # print(routing.id, min_v, max_v, mode, in_poly_idx)

            inner_routes.append([routing, section, mode])
            plot_points[routing.id] = [routing.wpt_list[idx].location.array() for idx in section]

    if visual:
        plot_line(vertices, plot_points, 'scripts/sector_with_routes')

    return inner_routes


# step 3
def get_fpl_random(routes, number=100):
    np.random.shuffle(fpl_set)

    assign_time_interval = 30
    count = 0

    fpl_list, starts = [], []
    for j in range(0, 1200, assign_time_interval):
        np.random.shuffle(routes)
        starts.append(j)

        for [route, section, mode] in routes[:number]:
            # routing
            routing = Routing(id=route.id, waypointList=route.get_points(section=section), other=section)

            # start time
            # start = np.random.randint(j, j+assign_time_interval)

            # min_alt, max_alt
            min_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            if np.random.randint(0, 60) % 4 == 0:
                max_alt = flight_level[np.random.randint(0, len(flight_level) - 1)]
            else:
                max_alt = min_alt

            if mode == 'start':  # 在扇区内起飞的航班，上升后平飞
                min_alt = 6000.0
            elif mode == 'end':  # 在扇区内落地的航班，下降
                max_alt = 6000.0

            # aircraft
            fpl = fpl_set[count]
            ac = fpl.aircraft

            # flight plan
            fpl = FlightPlan(id=fpl.id, routing=routing, aircraft=ac, startTime=j, min_alt=min_alt, max_alt=max_alt)
            print(count, fpl.id, ac.id, j, routing.id, len(routing.waypointList) <= 1, min_alt, max_alt)

            fpl_list.append(fpl)
            count += 1

    return fpl_list, starts


# step 4
def run_scenario(fpl_list, starts, video=None):
    print('>>>', len(fpl_list))
    agent_set = AircraftAgentSet(fpl_list=fpl_list, starts=starts)

    conflicts_dict, record = {}, {'flow': [], 'conflict': []}
    shift_list = []

    start = time.time()
    while True:
        agent_set.do_step(duration=8)

        conflicts = []
        for c in agent_set.detect_conflict_list():
            [a0, a1] = c.id.split('-')
            fpl0 = agent_set.agents[a0].fpl
            fpl1 = agent_set.agents[a1].fpl

            ok = False
            if c.time - fpl0.startTime < shift:
                shift_list.append(a0)
                ok = True
            if c.time - fpl1.startTime < shift:
                shift_list.append(a1)
                ok = True
            if ok:
                print('-------------------------------------')
                print('|  Conflict ID: ', c.id)
                print('|Conflict Time: ', c.time)
                print('|   H Distance: ', c.hDist)
                print('|   V Distance: ', c.vDist)
                print('|     a0 state: ', c.pos0)
                print('|      a0 info: ', fpl0.startTime, fpl0.min_alt, fpl0.max_alt, fpl0.routing.id)
                print('|     a1 state: ', c.pos1)
                print('|      a1 info: ', fpl1.startTime, fpl1.min_alt, fpl1.max_alt, fpl0.routing.id)
                print('-------------------------------------')
            conflicts.append(c)

        if len(conflicts) > 0:
            conflicts_dict[agent_set.time] = conflicts

        record['flow'].append(len(agent_set.agent_en))
        record['conflict'].append(len(conflicts))

        if agent_set.done():
            print('场景运行结束：', agent_set.time, len(conflicts), time.time() - start)
            break

    return conflicts, record, shift_list


# Step 7
def write_in_db(name, conflict_info, fpl_info):
    database = pymongo.MongoClient('localhost')['admin']
    collection = database['scenarios_600_5000']

    conflict_list = [c.to_dict() for [c, *_] in conflict_info]
    fpl_list = [fpl.to_dict() for fpl in fpl_info]
    collection.insert_one(dict(id=name, conflict_list=conflict_list, fpl_list=fpl_list))


def main():
    np.random.seed(1234)
    inner_routes = search_routing_in_wuhan(visual=False)
    print('>>> 一共找到{}条经过武汉扇区的Routing（Step 1和2）\n'.format(len(inner_routes)))

    for i in range(0, 5):
        fpl_list, starts = get_fpl_random(inner_routes[:], number=20)
        print('>>> 随机加载航空器（Step 3）\n')

        print('>>> 开始运行场景，并进行冲突探测（Step 4和5）')
        conflicts, record, shift_list = run_scenario(fpl_list, starts)

        new_fpl_list, new_starts = [], []
        for fpl in fpl_list:
            if fpl.id not in shift_list:
                new_fpl_list.append(fpl)
                new_starts.append(fpl.startTime)
                print(fpl.to_dict())

        print('>>> 重新运行场景，并进行冲突探测（Step 6和5）')
        new_conflicts, new_record, shift_list = run_scenario(new_fpl_list, new_starts, video='scripts/after')
        assert len(shift_list) == 0

        print('>>> 记录冲突信息和飞行计划（Step 7）')
        # write_in_db(i, new_conflicts, new_fpl_list)

        fig, axes = plt.subplots(3, 1)

        axes[0].plot(record['flow'], label='before_flow')
        axes[0].plot(new_record['flow'], label='after_flow')
        # axes[0].set_xticks(list(range(0, 301, 20)))
        axes[0].set_xlabel('Time Axis/30 second')
        axes[0].set_ylabel('Flow')
        axes[0].legend()

        axes[1].plot(record['conflict'], label='before_conflict')
        axes[1].plot(new_record['conflict'], label='after_conflict')
        # axes[1].set_xticks(list(range(0, 301, 20)))
        axes[1].set_xlabel('Time Axis/30 second')
        axes[1].set_ylabel('Conflict')
        axes[1].legend()

        axes[2].hist([starts, new_starts], bins=list(range(min_time, max_time, 600)),
                     label=['before_starts', 'after_starts'])
        axes[2].set_xlabel('Time Axis/second')
        axes[2].set_ylabel('Start')
        axes[2].legend()

        plt.subplots_adjust(hspace=0.5)
        fig.savefig('scripts/tmp.pdf')
        plt.show()
        break


if __name__ == '__main__':
    main()
