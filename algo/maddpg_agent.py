import random
from copy import deepcopy

from torch.optim import Adam
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from algo.network import Critic, Actor
from algo.memory import ReplayMemory, Experience
from algo.misc import *


class MADDPG:
    def __init__(self, dim_obs, dim_act, args, **kwargs):
        self.args = args
        self.var = 1.0

        self.actor = Actor(dim_obs, dim_act).to(device)
        self.critic = Critic(dim_obs, dim_act).to(device)
        self.actor_target = deepcopy(self.actor).to(device)
        self.critic_target = deepcopy(self.critic).to(device)

        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.c_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.a_lr)

        self.memory = ReplayMemory(args.memory_length)
        self.c_loss, self.a_loss = [], []

        self.__addiction(dim_obs, dim_act, **kwargs)

    def __addiction(self, dim_obs, dim_act, log_path=None, graph_path=None, load_path=None):
        if log_path is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(log_path)

        if graph_path is not None:
            print('Draw the net of Actor and Critic!')
            net_visual([(1, dim_obs)], self.actor,
                       filename='actor', directory=graph_path, format='png', cleanup=True)
            net_visual([(1, 2, dim_obs), (1, 2, dim_act)], self.critic,
                       filename='critic', directory=graph_path, format='png', cleanup=True)

        if load_path is not None:
            self.load_model(load_path)
            print("Load model successfully!")

    def load_model(self, load_path):
        [path, episode] = load_path
        actor = th.load(path + "actor_{}.pth".format(episode))
        critic = th.load(path + "critic_{}.pth".format(episode))

        self.actor.load_state_dict(actor.state_dict())
        self.critic.load_state_dict(critic.state_dict())
        self.actor_target = deepcopy(self.actor)
        self.critic_target = deepcopy(self.critic)

    def save_model(self, save_path, episode):
        th.save(self.actor, save_path + 'actor_{}.pth'.format(episode))
        th.save(self.critic, save_path + 'critic_{}.pth'.format(episode))

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def update(self, step, step_size):
        for n_agent, experiences in self.memory.sample(self.args.batch_size, num_iter=self.args.inner_iter):
            for e in experiences:
                batch = Experience(*zip(*e))

                state_batch = th.stack(batch.states).type(FloatTensor)
                action_batch = th.stack(batch.actions).type(FloatTensor)
                reward_batch = th.stack(batch.reward).type(FloatTensor)
                next_states = th.stack(batch.next_states).type(FloatTensor)

                # 更新Critic
                self.critic.zero_grad()
                self.critic_optimizer.zero_grad()

                current_q = self.critic(state_batch, action_batch)
                target_q = reward_batch

                q_loss = nn.MSELoss()(current_q, target_q.detach())
                q_loss.backward()
                th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
                self.critic_optimizer.step()
                self.c_loss.append(q_loss.item())

                # 更新Actor
                self.actor.zero_grad()
                self.actor_optimizer.zero_grad()

                actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                self.actor_optimizer.step()
                self.a_loss.append(actor_loss.item())

        if step > 0 and step % 100 == 0:
            soft_update(self.critic_target, self.critic, self.args.tau)
            soft_update(self.actor_target, self.actor, self.args.tau)

            self.writer.add_scalars('L', {'critic': np.mean(self.c_loss),
                                          'actor': np.mean(self.a_loss),
                                          'step': step_size}, step)
            self.c_loss, self.a_loss = [], []

    # def update(self, step, step_size):
    #     actor_old_vars = self.actor.state_dict()
    #     # critic_old_vars = self.critic.state_dict()
    #
    #     actor_new_vars, critic_new_vars = [], []
    #     for n_agent, transitions in self.memory.sample(self.args.batch_size, num_iter=self.args.inner_iter):
    #         self.actor.load_state_dict(actor_old_vars)
    #         # self.critic.load_state_dict(critic_old_vars)
    #
    #         for transition in transitions:
    #             batch = Experience(*zip(*transition))
    #
    #             state_batch = th.stack(batch.states).type(FloatTensor).to(device)
    #             action_batch = th.stack(batch.actions).type(FloatTensor).to(device)
    #             reward_batch = th.stack(batch.reward).type(FloatTensor).to(device)
    #             next_states = th.stack(batch.next_states).type(FloatTensor).to(device)
    #
    #             # 更新Critic
    #             self.critic.zero_grad()
    #             self.critic_optimizer.zero_grad()
    #
    #             current_q = self.critic(state_batch, action_batch)
    #             next_actions = self.actor_target(next_states)
    #             target_q = self.critic_target(next_states, next_actions)
    #             target_q = target_q * self.args.gamma + reward_batch
    #
    #             q_loss = nn.MSELoss()(current_q, target_q.detach())
    #             q_loss.backward()
    #             th.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
    #             self.critic_optimizer.step()
    #             self.c_loss.append(q_loss.item())
    #
    #             # 更新Actor
    #             self.actor.zero_grad()
    #             self.actor_optimizer.zero_grad()
    #
    #             actor_loss = -self.critic(state_batch, self.actor(state_batch)).mean()
    #             actor_loss.backward()
    #             th.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
    #             self.actor_optimizer.step()
    #             self.a_loss.append(actor_loss.item())
    #
    #         actor_new_vars.append(self.actor.state_dict())
    #         # critic_new_vars.append(self.critic.state_dict())
    #
    #     self.actor.load_state_dict(interpolate_vars(actor_old_vars, actor_new_vars, step_size))
    #     # self.critic.load_state_dict(interpolate_vars(critic_old_vars, critic_new_vars, step_size))
    #
    #     if step > 0 and step % 100 == 0:
    #         soft_update(self.critic_target, self.critic, self.args.tau)
    #         soft_update(self.actor_target, self.actor, self.args.tau)
    #
    #         self.writer.add_scalars('L', {'c': np.mean(self.c_loss), 'a': np.mean(self.a_loss), 's': step_size}, step)
    #         self.c_loss, self.a_loss = [], []

    def choose_action(self, states, noisy=True):
        states = th.from_numpy(states).float().to(device)

        actions, rand = self.actor(states), False
        if noisy and random.random() <= self.var:
            actions += th.randn_like(actions).type(FloatTensor).to(device)
            actions = th.clamp(actions, -1, 1)
            rand = True

        if self.var > 0.05:
            self.var *= 0.99995

        return actions.data.cpu().numpy(), rand
