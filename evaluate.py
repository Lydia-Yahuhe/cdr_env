import csv
from copy import deepcopy
import cv2

from train import *
from env.utils import border_func
from algo.misc import *

standard_width = 1600
standard_height = 900


class NetLooker:
    def __init__(self, net, name='net', width=16000, height=9000, channel=3, ratio=0.5, **kwargs):
        self.net = net
        self.vars = net.state_dict()
        self.name = name

        self.width = width
        self.height = height
        self.channel = channel
        self.ratio = ratio

        self.__look_weights_and_biases(**kwargs)

    def __look_weights_and_biases(self, look_weight=False, look_bias=True, scale=10):
        # 创建一个的黑色画布，RGB(0,0,0)即黑色
        image = np.ones((self.height, self.width, self.channel), np.uint8) * 255

        length = len(self.vars)

        self.rows = length // 2 + 1
        c_start = int(self.height * 0.05)  # 最左侧节点圆心与左侧边界的距离
        r_interval = (self.height - 2 * c_start) // (self.rows - 1)  # 相邻层节点圆心的距离
        c_interval = int(self.width // max([var.shape[-1] for var in self.vars.values()]))  # 同层相邻节点圆心的距离
        self.radius = int(c_interval * 0.4)

        var_list, link_metrix, count = [], [], 0
        for i, (var_name, var) in enumerate(self.vars.items()):
            var_list.append(var)
            if i % 2 != 0 and i != length - 1:
                continue

            dim_nodes = var.shape[-1]

            if count == 0:
                mode = 'input'
                string = "{}({})".format(mode, dim_nodes)
            elif count == self.rows - 1:
                mode = 'output'
                string = "{}({},{})".format(mode, dim_nodes, 'Tanh')
            else:
                mode = 'hidden'
                string = "{}({},{})".format(mode, dim_nodes, 'Relu')

            image = cv2.putText(image,
                                string,
                                (self.width // 2, c_start + count * r_interval + 10 * self.radius),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                self.width / standard_width * self.ratio, (255.0, 0, 0),
                                thickness=self.radius // 2)

            # 最上层节点圆心与上层边界的距离
            r_start = int(self.width // 2 - (dim_nodes - 1) / 2 * c_interval)

            coord_list = []
            for j in range(dim_nodes):
                coord = (r_start + j * c_interval, c_start + count * r_interval)
                coord_list.append(coord)
            link_metrix.append(coord_list)

            count += 1

        for i in range(len(var_list) // 2):
            weights, biases = var_list[i * 2], var_list[i * 2 + 1]
            print(weights.shape, len(link_metrix[i]), len(link_metrix[i + 1]))

            for j, coord1 in enumerate(link_metrix[i]):
                for k, coord2 in enumerate(link_metrix[i + 1]):
                    # bias的颜色（越大越长，负下正上）
                    if j == len(link_metrix[i + 1]) - 1 and look_bias:
                        c_bias = 255 if abs(biases[k] * scale) >= 1.0 else 150
                        image = cv2.line(image,
                                         coord2,
                                         (coord2[0],
                                          coord2[1] - border_func(biases[k] * scale, d_type=int) * 5 * self.radius),
                                         (0, 0, c_bias),
                                         thickness=self.radius // 2)

                    # weight连线的颜色（带符号的权重越大，越接近黑色）
                    if look_weight:
                        c_weight = (1.0 - border_func(abs(weights[k, j]), d_type=float)) * 255.0
                        image = cv2.line(image,
                                         (coord1[0], coord1[1] + self.radius), (coord2[0], coord2[1] - self.radius),
                                         (c_weight, c_weight, c_weight))

        self.link_matrix = link_metrix
        self.image = image

    def __look_layer(self, image, values, matrix):
        assert len(matrix) == len(values)

        for i, v in enumerate(values):
            thick = self.radius // 2 if v < 0 else -1
            c_node = (1 - border_func(abs(v), d_type=float)) * 255.0
            image = cv2.circle(image, matrix[i], self.radius, (c_node, c_node, c_node), thickness=thick)
        return image

    def look(self, inputs, folder=None, suffix=0, name='1_1.png'):
        ratio = self.ratio
        net = self.net

        image = deepcopy(self.image)

        result = th.from_numpy(inputs).float().to(device)
        for i in range(self.rows):
            if i == 0:
                result = result
            elif i == 1:
                result = F.relu(net.fc1(result))
            elif i == 2:
                result = F.relu(net.fc2(result))
            else:
                result = th.tanh(net.fc3(result))

            image = self.__look_layer(image, result[0, :], self.link_matrix[i])

        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)

        cv2.imshow(self.name, image)
        if cv2.waitKey(0) == 113:
            if folder is not None:
                path = os.path.join(folder, '{}_{}'.format(self.name, suffix))
                if not os.path.exists(path):
                    os.makedirs(path)

                cv2.imwrite(path + '/' + name, image)

    def close(self):
        cv2.waitKey(1) & 0xFF
        cv2.destroyAllWindows()


def test(suffix=0):
    args = args_parse()

    path = get_folder(make_exp_id(args), allow_exist=True)

    env = ConflictEnv(ratio=1.0, x=args.x, A=args.A, c_type=args.c_type)

    model = MADDPG(dim_obs=env.observation_space.shape[0],
                   dim_act=env.action_space.n,
                   args=args,
                   # load_path=['trained/pretrain/', suffix],
                   load_path=[path['model_path'], suffix])

    a_looker = NetLooker(net=model.actor, name='actor', look_weight=True)

    # 每百步的解脱率、每百回合的解脱率、每回合的步数
    record = []

    # 回合数、回合内步数、回合内奖励和、是否更换新的场景
    episode, t, change = 1, 0, True
    while not (env.eval_is_over() and change):
        states, done = env.reset(change=change), False

        # 如果states是None，则该回合的所有冲突都被成功解脱
        if states is not None:
            # a_looker.look(inputs=states,
            #               folder=path['graph_path'],
            #               suffix=suffix,
            #               name='{}_{}.png'.format(episode, t))

            actions, _ = model.choose_action(states, noisy=False)
            next_states, reward, done, info = env.step(actions)
            # env.render(counter='{}_{}'.format(t, episode))
            # states = next_states

            obs_tensors = th.from_numpy(states).float().to(device)
            act_tensors = th.from_numpy(actions).float().to(device)
            q = model.critic(obs_tensors.unsqueeze(0), act_tensors.unsqueeze(0))

            t += 1
            record.append([episode, t, int(states is None), int(done), reward, q[0][0].item()])
            print('{:>2d} {:>6d} {:>+4.2f} {:>+4.2f}'.format(t, episode, reward, q[0][0]))

        # 如果前个冲突成功解脱，则进入下一个冲突时刻，否则更换新的场景
        if not done:
            change = True
            t = 0
            episode += 1
        else:
            change = False

    a_looker.close()

    with open(os.path.join(path['folder'], 'record_{}.csv'.format(suffix)), 'w', newline='') as f:
        f = csv.writer(f)
        f.writerow(['info_id', 'episode', 't', 'sr_epi', 'sr_t', 'rew', 'q'])
        f.writerows(record)


if __name__ == '__main__':
    for epi in range(5000, 20001, 5000):
        test(epi)
