from typing import Dict, List

import numpy as np
import gym
from gym import spaces

from .core import CmdCount, AircraftAgent, AircraftAgentSet, parse_cmd
from .load import load_and_split_data
from .model import Conflict
from .render import CVRender, border_sector as border
from .utils import in_which_block


class ConflictScenario:
    def __init__(self, info, x=0, A=1, c_type='conc'):
        print('--------scenario------------', info['id'], x, A, c_type)
        self.info: Dict[str, object] = info

        self.x: int = x  # 变量1：时间范围大小
        self.A: int = A  # 变量2：空域范围（4——1/4空域）
        self.c_type: str = c_type  # 冲突解脱类型（conc——同时解脱，pair——两两解脱）

        self.shin: AircraftAgentSet = AircraftAgentSet(fpl_list=info['fpl_list'], candi=info['candi'])
        self.shin.step(1)
        self.kage: AircraftAgentSet = AircraftAgentSet(other=self.shin)  # 真：しん（shin），影：かげ（kage）
        self.kage.step(300)

        self.conflict_acs_seq: List[List[str]] = []
        self.conflict_acs: List[str] = []

        self.record: Dict[str, object] = {}

    def now(self) -> int:
        return self.shin.time

    def __get_conflict_ac(self, conflicts: List[Conflict]):
        if len(conflicts) == 1:
            return [conflicts[0].id.split('-')]

        if self.c_type == 'pair':  # 解脱方式为pair（两两解脱）
            conflict_acs = []
            check = []
            for c in conflicts:
                two = []
                [a0, a1] = c.id.split('-')
                if a0 not in check:
                    two.append(a0)
                    check.append(a0)
                if a1 not in check:
                    two.append(a1)
                    check.append(a1)
                conflict_acs.append(two)
        elif self.c_type == 'conc':  # 解脱方式为conc（同时解脱）
            conflict_acs = [[] for _ in range(self.A)]
            for c in conflicts:
                [a0, a1] = c.id.split('-')
                idx = in_which_block(c.pos0, border, self.A)
                conflict_acs[idx].append(a0)
                idx = in_which_block(c.pos1, border, self.A)
                conflict_acs[idx].append(a1)
        else:
            raise NotImplementedError

        return [list(set(lst)) for lst in conflict_acs if len(lst) > 0]

    def next(self, duration=5):
        if len(self.conflict_acs_seq) > 0:
            self.conflict_acs = self.conflict_acs_seq.pop(0)
            assert len(self.conflict_acs) >= 0
            return self.__get_states(a_set0=self.shin, a_set1=self.kage)

        while True:
            self.shin.step(duration)
            self.kage.step(duration)

            conflicts = self.kage.detect()
            if len(conflicts) <= 0:
                if self.shin.is_done():
                    return None
                continue

            if self.x > 0 and self.c_type == 'conc':
                ghost = AircraftAgentSet(other=self.kage)
                for i in range(self.kage.time + self.x):
                    ghost.step(duration)
                    conflicts += ghost.detect()

            self.record['r_conflicts'] = conflicts
            self.conflict_acs_seq = self.__get_conflict_ac(conflicts)
            self.conflict_acs = self.conflict_acs_seq.pop(0)

            return self.__get_states(a_set0=self.shin, a_set1=self.kage)

    def step(self, actions):
        shin_copy = AircraftAgentSet(other=self.shin)
        now = self.now()
        # 解析、分配动作
        cmd_info = self.__assign_cmd(now + 30, actions, agents=shin_copy.agents)
        # 检查动作的解脱效果，并返回下一部状态
        is_solved, next_states = self.__check_cmd_effect(now, a_set=shin_copy)
        # 根据指令和解脱效果，计算奖励
        reward = self.__calc_reward(is_solved, cmd_info, operator=min)
        return next_states, reward, is_solved, {}

    def __get_states(self, a_set0: AircraftAgentSet, a_set1: AircraftAgentSet):
        return np.concatenate(
            [
                a_set0.get_states(self.conflict_acs),
                a_set1.get_states(self.conflict_acs)
            ],
            axis=1
        )

    def __assign_cmd(self, t: int, actions: np.array, agents: Dict[str, AircraftAgent]):
        ret = {}
        for i, ac in enumerate(self.conflict_acs):
            cmd_list = parse_cmd(t, actions[i])
            agents[ac].assign_cmd(cmd_list)
            ret[ac] = cmd_list
        self.record['cmd_info'] = ret
        return ret

    def __check_cmd_effect(self, now: int, a_set: AircraftAgentSet):
        a_set_copy = AircraftAgentSet(other=a_set)

        tracks = {}
        conflicts = []
        is_solved = True
        while True:
            clock = a_set_copy.time
            tracks[clock] = a_set_copy.step(5)
            if clock == now + 300:
                self.ghost = AircraftAgentSet(other=a_set_copy)

            conflicts = a_set_copy.detect(search=self.conflict_acs)
            if len(conflicts) > 0:
                is_solved = False
                break

            if clock < now + 2 * 300:
                break

        if is_solved:
            self.agent_set = a_set
        self.record.update({'result': is_solved,
                            'tracks': tracks,
                            'f_conflicts': conflicts})
        return is_solved, self.__get_states(a_set0=self.kage, a_set1=a_set)

    def __calc_reward(self, solved: bool, cmd_info, operator=None):
        rewards = []

        for ac in self.conflict_acs:
            if solved:
                [alt_cmd, hdg_cmd, spd_cmd] = cmd_info[ac]
                rew_alt = 0.3 - abs(alt_cmd.delta) / 3000.0
                rew_hdg = 0.4 - abs(hdg_cmd.delta) / 150.0
                rew_spd = 0.3 - abs(spd_cmd.delta) / 100.0
                rewards.append(rew_alt + rew_spd + rew_hdg)
            else:
                rewards.append(-1.0)
        return rewards if operator is None else operator(rewards)


class ConflictEnv(gym.Env):
    def __init__(self, size=None, ratio=0.8, density=1.0, **kwargs):
        self.kwargs = kwargs
        self.train_set, self.test_set = load_and_split_data(size=size,
                                                            ratio=ratio,
                                                            density=density)
        self.action_space = spaces.Discrete(CmdCount)
        self.observation_space = spaces.Box(low=-1.0,
                                            high=+1.0,
                                            shape=(120,))
        print('----------env------------')
        print('|   split ratio: {:<6.2f} |'.format(ratio))
        print('|    train size: {:<6} |'.format(len(self.train_set)))
        print('| validate size: {:<6} |'.format(len(self.test_set)))
        print('|  action shape: {}   |'.format((self.action_space.n,)))
        print('|   state shape: {} |'.format(self.observation_space.shape))
        print('-------------------------')

        self.scenario = None
        self.idx = -1
        self.cv_render = CVRender(video_path='env/data/scenario.avi',
                                  image_path='env/data/wuhan_base_render.png')

    def reset(self, change=True):
        if change:
            self.idx = (self.idx + 1) % len(self.train_set)
            self.scenario = ConflictScenario(self.train_set[self.idx], **self.kwargs)
        return self.scenario.next()

    def step(self, actions):
        return self.scenario.step(actions)

    def render(self, mode='human', wait=1):
        self.cv_render.render(self.scenario, wait=wait)

    def close(self):
        self.cv_render.close()
