import pymongo
from tqdm import tqdm

from env.model import *


def dict_to_type(e):
    """
    加载性能数据表格（各个高度对应的标称速度、转弯率等）
    """
    ret = {}
    for key, values in e.items():
        if key != 'flightPerformanceTable':
            ret[key] = values
        else:
            ret[key] = [Performance(**v) for v in values]
    return AircraftType(**ret)


def load_type(database):
    """
    加载机型数据（最大加减速度和性能数据表等）
    """
    ret = {}
    for e in database['AircraftType'].find():
        del e['_id']
        key = e['id']
        ret[key] = dict_to_type(e)
    return ret


def load_waypoint(database):
    """
    加载航路点数据
    """
    ret = {}
    for e in database['Waypoint'].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['point']['lng'], e['point']['lat']))
    for e in database["Airport"].find():
        key = e['id']
        ret[key] = Waypoint(id=key,
                            location=Point2D(e['location']['lng'], e['location']['lat']))
    return ret


def load_routing(database, wpt_dict):
    """
    加载航线数据（城市OD对和计划航路点集合）
    """
    ret = {}
    for e in database['Routing'].find():
        key = e['id']
        ret[key] = Routing(id=key,
                           wpt_list=[wpt_dict[wpt_id] for wpt_id in e['waypointList']])
    return ret


def load_aircraft(database, act_dict):
    """
    加载航空器数据（注册号和机型）
    """
    ret = {}
    for e in database['AircraftRandom'].find():
        key = e['id']
        ret[key] = Aircraft(id=key,
                            aircraftType=act_dict[e['aircraftType']])
    return ret


def load_flight_plan(database, aircraft, routes):
    """
    加载航班的飞行计划数据（呼号、起始高度、航线、起飞时刻、航空器和目标高度）
    """
    ret = {}
    for e in database['FlightPlan'].find():
        key = e['id']
        ret[key] = FlightPlan(id=key,
                              min_alt=0,
                              routing=routes[e['routing']],
                              startTime=e['startTime'],
                              aircraft=aircraft[e['aircraft']],
                              max_alt=e['flightLevel'])
    return ret


def load_data_set(host='localhost', db='admin'):
    """
    包含所有飞行数据的集合
    """
    connection = pymongo.MongoClient(host)
    database = connection[db]
    wpt_dict = load_waypoint(database)
    act_dict = load_type(database)
    air_dict = load_aircraft(database, act_dict)
    rou_dict = load_routing(database, wpt_dict)
    fpl_dict = load_flight_plan(database, air_dict, rou_dict)
    connection.close()
    return DataSet(wpt_dict=wpt_dict,
                   rou_dict=rou_dict,
                   fpl_dict=fpl_dict,
                   air_dict=air_dict,
                   act_dict=act_dict)


def load_and_split_data(host='localhost', db='admin', col='scenarios', size=None, ratio=0.8, density=1):
    """
    加载冲突场景，并将其按比例分为训练集和测试集
    """
    connection = pymongo.MongoClient(host)
    database = connection[db]
    wpt_dict = load_waypoint(database)
    act_dict = load_type(database)
    rou_dict = load_routing(database, wpt_dict)

    if size is None:
        size = int(1e6)

    scenarios, count = [], 0
    for info in tqdm(database[col].find(), desc='Loading from {}/{}'.format(db, col)):
        fpl_list, candi = [], {}
        for i, fpl in enumerate(info['fpl_list']):
            if i % density != 0:
                continue

            # call sign
            fpl_id = fpl['id']
            # start time
            start = fpl['startTime']
            # routing
            routing = rou_dict[fpl['routing']].copy(section=fpl['other'])
            # aircraft
            ac = Aircraft(id=fpl['aircraft'], aircraftType=act_dict[fpl['acType']])
            # flight plan
            fpl = FlightPlan(id=fpl_id,
                             routing=routing,
                             aircraft=ac,
                             startTime=start,
                             min_alt=fpl['min_alt'],
                             max_alt=fpl['max_alt'])
            fpl_list.append(fpl)

            if start in candi.keys():
                candi[start].append(fpl_id)
            else:
                candi[start] = [fpl_id]

        scenarios.append(dict(id=str(count+1), fpl_list=fpl_list, candi=candi))
        count += 1
        if count >= size:
            break
    connection.close()
    split_size = int(count * ratio)
    return scenarios[:split_size], scenarios[split_size:]


if __name__ == '__main__':
    load_data_set()
