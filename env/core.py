from __future__ import annotations
from dataclasses import dataclass

from typing import List, Dict, Tuple

import numpy as np
from rtree.index import Index, Property

from .utils import make_bbox_3d, distance, position_in_bbox, calc_level, calc_turn_prediction, border_float
from .model import Point2D, FlightPlan, Routing, Waypoint, Performance, AircraftType, Conflict

CmdCount = 6


# ---------
# Command
# ---------
@dataclass
class ATCCmd:
    delta: float = 0.0
    assignTime: int = 0.0
    ok: bool = True
    cmdType: str = ""

    def to_dict(self):
        return {self.cmdType: '{},{}'.format(round(self.delta, 2), self.assignTime)}

    def __str__(self):
        return '%s: <TIME:%d, DELTA:%0.2f>' % (self.cmdType, self.assignTime, self.delta)

    def __repr__(self):
        return '%s: <TIME:%d, DELTA:%0.2f>' % (self.cmdType, self.assignTime, self.delta)


def parse_cmd(now: int, cmd: list):
    # alt cmd
    span: int = int((cmd[0] + 1) * 100)
    idx: float = round(cmd[1] * 3)
    alt_cmd = ATCCmd(delta=idx * 300.0, assignTime=now + span, cmdType='Altitude')
    print('{:>3d} {:>+3.1f}'.format(span, idx), end='\t')
    # hdg cmd
    span = int((cmd[2] + 1) * 100)
    idx = cmd[3] * 4
    hdg_cmd = ATCCmd(delta=idx * 15.0, assignTime=now + span, cmdType='Heading')
    print('{:>3d} {:>+3.1f}'.format(span, idx), end='\t')
    # spd cmd
    span = int((cmd[4] + 1) * 100)
    idx = cmd[5] * 3
    spd_cmd = ATCCmd(delta=idx * 10, assignTime=now + span, cmdType='Speed')
    print('{:>3d} {:>+3.1f}'.format(span, idx), end='\t')
    return [alt_cmd, hdg_cmd, spd_cmd]


# ---------
# Control
# ---------
@dataclass
class FlightControl(object):
    altCmd: ATCCmd = None
    spdCmd: ATCCmd = None
    hdgCmd: ATCCmd = None

    def __init__(self, fpl: FlightPlan):
        self.targetAlt: float = fpl.max_alt
        self.targetHSpd: float = 0.0
        self.targetCourse: float = 0.0

    def assign(self, cmd_list: List[ATCCmd]):
        for cmd in cmd_list:
            if cmd.cmdType == "Altitude":
                self.altCmd = cmd
            elif cmd.cmdType == "Speed":
                self.spdCmd = cmd
            elif cmd.cmdType == "Heading":
                self.hdgCmd = cmd
            else:
                raise NotImplementedError

    def update(self, now: int, v_spd: float, performance: Performance, alt: float, hdg: float,
               hdg_to_target: float):
        self.__update_target_spd(now, v_spd, performance)
        self.__update_target_alt(now, v_spd, alt)
        self.__update_target_hdg(now, hdg, hdg_to_target)

    def __update_target_alt(self, now: int, v_spd: float, alt: float):
        alt_cmd = self.altCmd
        if alt_cmd is None or now != alt_cmd.assignTime:
            return
        delta: float = alt_cmd.delta
        target_alt: float = calc_level(alt, v_spd, delta)
        if v_spd * delta >= 0.0 and 12000.0 >= target_alt >= 6000.0:
            self.targetAlt = target_alt
        self.altCmd = None

    def __update_target_spd(self, now: int, v_spd: float, performance: Performance):
        spd_cmd = self.spdCmd
        if spd_cmd is None or now != spd_cmd.assignTime:
            if v_spd == 0.0:
                self.targetHSpd = performance.normCruiseTAS
            elif v_spd > 0.0:
                self.targetHSpd = performance.normClimbTAS
            else:
                self.targetHSpd = performance.normDescentTAS
            return
        self.spdCmd = None

    def __update_target_hdg(self, now: int, heading: float, hdg_to_target: float):
        hdg_cmd = self.hdgCmd
        if hdg_cmd is None:
            self.targetCourse = hdg_to_target
            return

        diff: int = now - hdg_cmd.assignTime
        if diff < 0:
            self.targetCourse = hdg_to_target
            return

        delta: float = hdg_cmd.delta
        if delta == 0 or diff == 240:  # 结束偏置（dogleg机动）
            self.targetCourse = hdg_to_target
            self.hdgCmd = None
        elif diff == 0:  # 以delta角度出航
            self.targetCourse = (delta + heading) % 360
        elif diff == 120:  # 转向后持续60秒飞行，之后以30°角切回航路
            self.targetCourse = (-abs(delta) / delta * (abs(delta) + 30) + heading) % 360

    def set(self, other: FlightControl):
        self.altCmd = other.altCmd
        self.spdCmd = other.spdCmd
        self.hdgCmd = other.hdgCmd
        self.targetAlt = other.targetAlt
        self.targetHSpd = other.targetHSpd
        self.targetCourse = other.targetCourse


# ---------
# Profile
# ---------
@dataclass
class FlightLeg:
    start: Waypoint
    end: Waypoint
    distance: float = 0
    course: float = 0

    def __post_init__(self):
        self.distance = self.start.distance_to(self.end)
        self.course = self.start.bearing_to(self.end)

    def copy(self):
        return FlightLeg(self.start, self.end)


@dataclass
class FlightProfile:
    def __init__(self, fpl: FlightPlan):
        self.route: Routing = fpl.routing
        self.legs: List[FlightLeg] = self.__make_legs()
        self.idx: int = 0
        self.cur_leg: FlightLeg = self.legs[0]
        self.distToTarget: float = 0
        self.courseToTarget: float = 0

    def __make_legs(self) -> List[FlightLeg]:
        wpt_list = self.route.wpt_list
        return [FlightLeg(wpt_list[i], p) for i, p in enumerate(wpt_list[1:])]

    def update(self, h_spd: float, heading: float, performance: Performance, location: Point2D) -> bool:
        if self.__target_passed(h_spd, heading, performance):
            self.idx += 1
            self.cur_leg = self.__next_n_leg(0)
            if self.cur_leg is None:  # 如果curLeg是None，则飞行计划结束
                return False
            self.distToTarget = self.cur_leg.distance
            self.courseToTarget = self.cur_leg.course
        else:
            target: Point2D = self.cur_leg.end.location
            self.distToTarget = location.distance_to(target)
            self.courseToTarget = location.bearing_to(target)

        return True

    def __next_n_leg(self, n: int):
        idx: int = self.idx + n  # 若总共9个航段，则idx ∈ [0,8]
        return self.legs[idx] if idx <= len(self.legs) - 1 else None

    def __target_passed(self, h_spd: float, heading: float, performance: Performance) -> bool:
        dist: float = self.distToTarget
        if dist <= h_spd * 1:
            return True
        next_leg: FlightLeg = self.__next_n_leg(1)
        if next_leg is not None:
            return dist <= calc_turn_prediction(h_spd,
                                                self.cur_leg.course,
                                                next_leg.course,
                                                performance.normTurnRate)
        return 270 > (heading - self.courseToTarget) % 360 >= 90

    def set(self, other: FlightProfile):
        self.route = other.route
        self.legs = other.legs
        self.idx = other.idx
        self.cur_leg = other.cur_leg
        self.distToTarget = other.distToTarget
        self.courseToTarget = other.courseToTarget


# ---------
# Status
# ---------
class FlightStatus:
    def __init__(self, fpl: FlightPlan):
        self.alt: float = fpl.min_alt
        self.performance: Performance = Performance()
        self.acType: AircraftType = fpl.aircraft.aircraftType
        self.acType.compute_performance(self.alt, self.performance)
        self.hSpd: float = self.performance.normCruiseTAS
        self.vSpd: float = 0
        wpt_list: List[Waypoint] = fpl.routing.wpt_list
        self.location: Point2D = wpt_list[0].location.copy()
        self.heading: float = wpt_list[0].bearing_to(wpt_list[1])

    def update(self, target_h_spd: float, target_course: float, target_alt: float):
        self.__move_horizontal(target_h_spd, target_course)
        self.__move_vertical(target_alt)
        self.__update_performance()

    def __move_horizontal(self, target_h_spd: float, target_hdg: float):
        pre_h_spd: float = self.hSpd
        if pre_h_spd > target_h_spd:
            self.hSpd = max(pre_h_spd - self.acType.normDeceleration * 1, target_h_spd)
        elif pre_h_spd < target_h_spd:
            self.hSpd = min(pre_h_spd + self.acType.normAcceleration * 1, target_h_spd)
        diff: float = (target_hdg - self.heading) % 360
        diff = diff - 360 if diff > 180 else diff
        if abs(diff) >= 90:
            turn: float = self.performance.maxTurnRate * 1
        else:
            turn: float = self.performance.normTurnRate * 1
        diff = border_float(diff, -turn, turn)
        self.heading = (self.heading + diff) % 360
        self.location.move_to(self.heading, (pre_h_spd + self.hSpd) * 1 / 2)

    def __move_vertical(self, target_alt: float):
        diff: float = target_alt - self.alt
        if diff < 0:
            v_spd: float = max(-self.performance.normDescentRate * 1, diff)
        elif diff > 0:
            v_spd: float = min(self.performance.normClimbRate * 1, diff)
        else:
            v_spd: float = 0.0
        self.alt += v_spd
        self.vSpd = v_spd

    def __update_performance(self):
        self.acType.compute_performance(self.alt, self.performance)

    def get_state(self) -> Tuple[float, float, float, float, float, float]:
        return self.location.tuple() + (self.alt, self.hSpd, self.vSpd, self.heading)

    def get_position(self) -> Tuple[float, float, float]:
        return self.location.tuple() + (self.alt,)

    def set(self, other: FlightStatus):
        self.hSpd = other.hSpd
        self.vSpd = other.vSpd
        self.location.set(other.location)
        self.alt = other.alt
        self.heading = other.heading
        self.acType = other.acType
        self.performance.copy(other.performance)


# ---------
# Agent
# ---------
class AircraftAgent:
    def __init__(self, fpl):
        self.fpl: FlightPlan = fpl
        self.id: str = fpl.id
        self.phase: str = 'Schedule'
        self.control = FlightControl(fpl)
        self.status = FlightStatus(fpl)
        self.profile = FlightProfile(fpl)

    def is_enroute(self):
        return self.phase == 'EnRoute'

    def is_finished(self):
        return self.phase == "Finished"

    def state(self):
        return self.status.get_state()

    def position(self):
        return self.status.get_position()

    def step(self, now: int):
        if self.is_finished():  # 如果该航班飞行已经结束，则直接return
            return False

        if self.is_enroute():
            status, profile, control = self.status, self.profile, self.control
            control.update(now,
                           status.vSpd,
                           status.performance,
                           status.alt,
                           status.heading,
                           profile.courseToTarget)
            status.update(control.targetHSpd,
                          control.targetCourse,
                          control.targetAlt)
            if not profile.update(status.hSpd,
                                  status.heading,
                                  status.performance,
                                  status.location):
                self.phase = 'Finished'
        elif now == self.fpl.startTime:  # 如果当前时刻为起飞时刻，则状态改为EnRoute
            self.phase = 'EnRoute'

        return self.is_enroute()

    def assign_cmd(self, cmd_list: List[ATCCmd]):
        self.control.assign(cmd_list)

    def copy(self):
        other = AircraftAgent(self.fpl)
        other.control.set(self.control)
        other.status.set(self.status)
        other.profile.set(self.profile)
        return other


# ---------
# AgentSet
# ---------
class AircraftAgentSet:
    def __init__(self, fpl_list=None, candi=None, other=None):
        if fpl_list is not None:
            self.candi: dict = candi
            self.time: int = min(candi.keys()) - 1
            self.end: int = max(candi.keys())
            self.agents: Dict[str, AircraftAgent] = {fpl.id: AircraftAgent(fpl) for fpl in fpl_list}
            self.agent_id_en: List[str] = []
        else:
            self.candi: dict = other.candi
            self.time: int = other.time
            self.end: int = other.end
            self.agents: Dict[str, AircraftAgent] = {a_id: agent.copy() for a_id, agent in other.agents.items()}
            self.agent_id_en: List[str] = other.agent_id_en[:]

    def is_done(self) -> bool:
        return self.time > self.end and len(self.agent_id_en) <= 1

    def __pre_do_step(self, clock: int) -> List[str]:
        if clock in self.candi.keys():
            return self.agent_id_en + self.candi[clock]
        return self.agent_id_en

    def step(self, duration: int, basic=False):
        now = self.time
        duration -= now * int(basic)
        agents = self.agents

        points = []
        for i in range(duration):
            clock = now + i + 1
            agent_id_en = []
            points = []
            for agent_id in self.__pre_do_step(clock):
                agent = agents[agent_id]

                if agent.step(clock):
                    agent_id_en.append(agent_id)
                    points.append((agent_id, ) + agent.state())
            self.agent_id_en = agent_id_en

        self.time = now + duration
        return points

    def __build_rtree(self):
        agent_id_en = self.agent_id_en
        agents = self.agents

        idx = Index(properties=Property(dimension=3))
        for i, a_id in enumerate(agent_id_en):
            agent = agents[a_id]
            idx.insert(i, make_bbox_3d(agent.position(), ext=(0.0, 0.0, 0.0)))
        return idx, agents, agent_id_en

    def get_states(self, conflict_acs, length=20, ext=(0.5, 0.5, 900.0)):
        r_tree, agents, agent_id_en = self.__build_rtree()

        states = []
        for conflict_ac in conflict_acs:
            a0 = agents[conflict_ac]
            pos0 = a0.position()
            bbox = make_bbox_3d(pos0, ext)
            state_dict = {}
            for i in r_tree.intersection(bbox):
                a1 = agents[agent_id_en[i]]
                if a0 == a1:
                    continue
                pos1 = a1.position()
                state_dict[position_in_bbox(bbox, pos1, ext)] = [
                    pos1[0],
                    pos1[1],
                    pos1[2] / 300.0
                ]

            if len(state_dict) <= length:
                state = [[0.0 for _ in range(3)] for _ in range(length)]
                j = 0
                for key in sorted(state_dict.keys()):
                    state[j] = state_dict[key]
                    j += 1
            else:
                state = [state_dict[key] for key in sorted(state_dict.keys())]
                delta = len(state) - length
                if delta % 2 == 0:
                    state = state[int(delta / 2):-int(delta / 2)]
                else:
                    state = state[int((delta - 1) / 2):-int((delta + 1) / 2)]
            states.append(np.concatenate(state))
        return states

    def detect(self, search=None):
        r_tree, agents, agent_id_en = self.__build_rtree()

        if len(agent_id_en) <= 1:
            return []
        if search is None:
            search = agent_id_en

        conflicts = []
        check_list = []
        for a0_id in search:
            a0 = agents[a0_id]
            if not a0.is_enroute():
                continue

            pos0 = a0.position()
            bbox = make_bbox_3d(pos0, (0.1, 0.1, 299))
            for i in r_tree.intersection(bbox):
                a1_id = agent_id_en[i]

                c_id = a0_id + '-' + a1_id
                if a0_id == a1_id or c_id in check_list:
                    continue

                a1 = agents[a1_id]
                pos1 = a1.position()

                h_dist = distance(pos0[:2], pos1[:2])
                v_dist = abs(pos0[2] - pos1[2])
                if h_dist < 10000 and v_dist < 300.0:
                    conflicts.append(Conflict(id=c_id,
                                              time=self.time,
                                              hDist=h_dist,
                                              vDist=v_dist,
                                              fpl0=a0.fpl,
                                              fpl1=a1.fpl,
                                              pos0=pos0,
                                              pos1=pos1))
                check_list.append(a1_id + '-' + a0_id)
        return conflicts
