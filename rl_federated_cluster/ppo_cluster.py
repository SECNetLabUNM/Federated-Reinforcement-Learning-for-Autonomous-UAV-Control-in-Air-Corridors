import copy
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset
import sys

sys.path.append('/home/kun/PycharmProjects/air-corridor_ncfo/')
from rl_federated import net_nn_fc_10
from rl_multi_3d_trans import net_nn_fc_10_3e

net_models = {

    'fc10': net_nn_fc_10,
    'fc10_3e': net_nn_fc_10_3e
    # add more mappings as needed
}


class MyDataset(Dataset):
    def __init__(self, data, env_with_Dead=True):
        self.data = data
        self.env_with_Dead = env_with_Dead

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        transition = self.data[idx]
        s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw, td_target, adv = transition

        # If your environment does not include Dead, modify dw here
        if self.env_with_Dead:  # Replace with your condition
            dw = False

        return {
            's1': torch.tensor(s1, dtype=torch.float),
            's2': torch.tensor(s2, dtype=torch.float),
            'a': torch.tensor(a, dtype=torch.float),
            'r': torch.tensor([r], dtype=torch.float),
            's1_prime': torch.tensor(s1_prime, dtype=torch.float),
            's2_prime': torch.tensor(s2_prime, dtype=torch.float),
            'logprob_a': torch.tensor(logprob_a, dtype=torch.float),
            'done': torch.tensor([done], dtype=torch.float),
            'dw': torch.tensor([dw], dtype=torch.float),
            'td_target': torch.tensor(td_target, dtype=torch.float),
            'adv': torch.tensor(adv, dtype=torch.float),
        }


class PPO(object):
    def __init__(
            self,
            state_dim=26,
            s2_dim=22,
            action_dim=3,
            env_with_Dead=True,
            gamma=0.99,
            lambd=0.95,
            # gamma=0.89,
            # lambd=0.88,
            clip_rate=0.2,
            K_epochs=10,
            net_width=256,
            a_lr=3e-4,
            c_lr=3e-4,
            l2_reg=1e-3,
            dist='Beta',
            a_optim_batch_size=64,
            c_optim_batch_size=64,
            entropy_coef=0,
            entropy_coef_decay=0.9998,
            writer=None,
            activation=None,
            share_layer_flag=True,
            anneal_lr=True,
            totoal_steps=0,
            with_position=False,
            token_query=False,
            num_enc=5,
            num_dec=5,
            logger=None,
            dir=None,
            test=False,
            net_model='fc1',
            beta_base=1e-5,
            num_agents=6,
            fed_key='all',
            cluster=3
    ):
        self.cluster = cluster
        self.fed_key = fed_key
        self.dir = dir
        self.logger = logger
        self.share_layer_flag = share_layer_flag
        self.dist = dist
        self.env_with_Dead = env_with_Dead
        self.action_dim = action_dim
        self.clip_rate = clip_rate
        self.gamma = gamma
        self.lambd = lambd
        self.clip_rate = clip_rate
        self.K_epochs = K_epochs
        self.data = {}
        self.l2_reg = l2_reg
        self.a_optim_batch_size = a_optim_batch_size
        self.c_optim_batch_size = c_optim_batch_size
        self.entropy_coef = entropy_coef
        self.entropy_coef_decay = entropy_coef_decay
        self.writer = writer
        self.anneal_lr = anneal_lr
        self.totoal_steps = totoal_steps
        self.a_lr = a_lr
        self.c_lr = c_lr
        self.num_agents = num_agents

        # global model does not use shared model
        actor_merge = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                        with_position=with_position, token_query=token_query,
                                                        num_enc=num_enc, num_dec=num_dec)
        self.global_actor = net_models[net_model].BetaActorMulti(state_dim, s2_dim, action_dim, net_width, actor_merge,
                                                                 beta_base).to(device)
        critic_merge = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                         with_position=with_position, token_query=token_query,
                                                         num_enc=num_enc, num_dec=num_dec)
        self.global_critic = net_models[net_model].CriticMulti(state_dim, s2_dim, net_width, critic_merge).to(device)

        # local models can share locally
        self.local_actors = []
        self.local_critics = []

        for _ in range(self.cluster):
            actor_merge = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                            with_position=with_position, token_query=token_query,
                                                            num_enc=num_enc, num_dec=num_dec)
            if share_layer_flag:
                critic_merge = actor_merge
            else:
                critic_merge = net_models[net_model].MergedModel(s1_dim=state_dim, s2_dim=s2_dim, net_width=net_width,
                                                                 with_position=with_position, token_query=token_query,
                                                                 num_enc=num_enc, num_dec=num_dec)
            self.local_actors.append(
                net_models[net_model].BetaActorMulti(state_dim, s2_dim, action_dim, net_width, actor_merge,
                                                     beta_base).to(device))
            self.local_critics.append(
                net_models[net_model].CriticMulti(state_dim, s2_dim, net_width, critic_merge).to(device))
        self.local_actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=a_lr) for actor in self.local_actors]
        self.local_critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=c_lr) for critic in self.local_critics]

    def load_pretrained(self):
        pass

    def average_model_weights(self, models, for_saving=False):
        # Step 1: Get the structure of the state_dict from the first model
        avg_state_dict = OrderedDict({key: value.clone() for key, value in models[0].state_dict().items()})

        # Step 2: Sum the parameters across all models, skipping those without requires_grad
        for model in models[1:]:
            model_state_dict = model.state_dict()
            for key, value in model_state_dict.items():
                if value.requires_grad:
                    avg_state_dict[key] += value.clone()

        # Step 3: Average the parameters
        num_models = len(models)
        for key, value in model_state_dict.items():
            if value.requires_grad:
                avg_state_dict[key] /= num_models

        if for_saving:
            return avg_state_dict
        else:
            # Step 4: Assign the averaged state dictionary to each model
            # First, prepare a full state dictionary with non-trainable parameters untouched
            for model in models:
                model.load_state_dict(avg_state_dict)
    def fed_average(self):
        if self.fed_key not in ['all', 'actor', 'critic']:
            return
        elif self.fed_key == 'all':
            keys = ('critic', 'acotr')
        else:
            keys = [self.fed_key]
        for key in keys:
            if key == 'critic':
                self.average_model_weights(self.local_critics)

            if key == 'actor':
                self.average_model_weights(self.local_actors)
            # if key == 'critic':
            #     average_weights = self.average_model_weights(self.local_critics)
            #     self.global_critic.load_state_dict(average_weights)
            #     [critic.load_state_dict(average_weights) for critic in self.local_critics]
            # if key == 'actor':
            #     average_weights = self.average_model_weights(self.local_actors)
            #     self.global_actor.load_state_dict(average_weights)
            #     [actor.load_state_dict(average_weights) for actor in self.local_actors]

    def select_action(self, s1, s2, agent_index):  # only used when interact with the env
        model_index = agent_index % self.cluster
        self.local_actors[model_index].eval()
        with torch.no_grad():
            s1 = np.array(s1)
            s1 = torch.FloatTensor(s1).to(device)
            s2 = np.array(s2)
            s2 = torch.FloatTensor(s2).to(device)

            dist, alpha, beta = self.local_actors[model_index].get_dist(s1, s2)

            assert torch.all((0 <= alpha))
            assert torch.all((0 <= beta))
            a = dist.sample()
            assert torch.all((0 <= a)) and torch.all(a <= 1)
            a = torch.clamp(a, 0, 1)
            logprob_a = dist.log_prob(a).cpu().numpy()
            return a.cpu().numpy(), logprob_a, alpha, beta



    def train(self, global_step, epoches=None, syn_round=False):
        if self.anneal_lr:
            frac = 1.0 - global_step / self.totoal_steps
            alrnow = frac * self.a_lr
            clrnow = frac * self.c_lr
        #
        #     self.actor_optimizer.param_groups[0]["lr"] = alrnow
        #     self.critic_optimizer.param_groups[0]["lr"] = clrnow

        self.entropy_coef *= self.entropy_coef_decay

        transitions = self.gae()
        dataset = [MyDataset(transition) for transition in transitions]

        dataloader = [DataLoader(ds, batch_size=self.a_optim_batch_size, shuffle=True, drop_last=True) for ds in
                      dataset]

        clipfracs = []
        for j in range(self.cluster):

            '''update the actor-critic'''
            actor = self.local_actors[j]
            actor.train()
            actor_optimizer = self.local_actor_optimizers[j]

            critic = self.local_critics[j]
            critic.train()
            critic_optimizer = self.local_critic_optimizers[j]
            # if self.anneal_lr:
            #     actor_optimizer.param_groups[0]["lr"] = alrnow
            #     critic_optimizer.param_groups[0]["lr"] = clrnow

            for _ in range(epoches):
                for batch in dataloader[j]:
                    s1 = batch['s1'].to(device)
                    s2 = batch['s2'].to(device)
                    a = batch['a'].to(device)
                    logprob_a = batch['logprob_a'].to(device)
                    adv = batch['adv'].to(device)
                    td_target = batch['td_target'].to(device)

                    '''derive the actor loss'''
                    # distribution, _, _, nan_event = self.actor.get_dist(s1, s2, self.logger)
                    distribution, alpha, beta = actor.get_dist(s1, s2)
                    # if nan_event:
                    #     self.save('nan')
                    dist_entropy = distribution.entropy().sum(1, keepdim=True)
                    logprob_a_now = distribution.log_prob(a)

                    logratio = logprob_a_now.sum(1, keepdim=True) - logprob_a.sum(1, keepdim=True)
                    ratio = torch.exp(logratio)  # a/b == exp(log(a)-log(b))

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [torch.mean((ratio - 1.0).abs() > self.clip_rate, dtype=torch.float32).item()]

                    surr1 = ratio * adv
                    surr2 = torch.clamp(ratio, 1 - self.clip_rate, 1 + self.clip_rate) * adv
                    pg_loss = -torch.min(surr1, surr2)
                    a_loss = pg_loss - self.entropy_coef * dist_entropy

                    '''derive the critic loss'''
                    # index = slice(i * self.c_optim_batch_size, min((i + 1) * self.c_optim_batch_size, s1.shape[0]))
                    c_loss = (critic(s1, s2) - td_target).pow(2).mean()
                    for name, param in critic.named_parameters():
                        if 'weight' in name:
                            c_loss += param.pow(2).sum() * self.l2_reg

                    '''updata parameters'''
                    actor_optimizer.zero_grad()
                    a_loss.mean().backward(retain_graph=self.share_layer_flag)
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), 40)
                    actor_optimizer.step()
                    critic_optimizer.zero_grad()
                    c_loss.backward()
                    critic_optimizer.step()

        if syn_round:
            # default only average over critic, if shared net for merge, then the merge part is also averaged.
            self.fed_average()

        # self.writer.add_scalar("weights/critic_learning_rate", self.critic_optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", c_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.mean().item(), global_step)
        self.writer.add_scalar("losses/entropy", dist_entropy.mean().item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        del a_loss, c_loss, pg_loss, dist_entropy, old_approx_kl, approx_kl, logprob_a_now, logratio  # , perm
        del surr1, surr2
        torch.cuda.empty_cache()
        self.data = {}  # Clean history trajectory

    def make_batch(self, agent):
        s1_lst = []
        s2_lst = []
        a_lst = []
        r_lst = []
        s1_prime_lst = []
        s2_prime_lst = []
        logprob_a_lst = []
        done_lst = []
        dw_lst = []
        for transition in self.data[agent]:
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done, dw = transition
            s1_lst.append(s1)
            s2_lst.append(s2)
            a_lst.append(a)
            logprob_a_lst.append(logprob_a)
            r_lst.append([r])
            s1_prime_lst.append(s1_prime)
            s2_prime_lst.append(s2_prime)
            done_lst.append([done])
            dw_lst.append([dw])

        if not self.env_with_Dead:
            '''Important!!!'''
            # env_without_DeadAndWin: deltas = r + self.gamma * vs_ - vs
            # env_with_DeadAndWin: deltas = r + self.gamma * vs_ * (1 - dw) - vs
            dw_lst = (np.array(dw_lst) * False).tolist()

        '''list to tensor'''
        with torch.no_grad():
            s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask = \
                torch.tensor(np.array(s1_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(r_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s1_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(s2_prime_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(logprob_a_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(done_lst), dtype=torch.float).to(device), \
                    torch.tensor(np.array(dw_lst), dtype=torch.float).to(device),
        return s1, s2, a, r, s1_prime, s2_prime, logprob_a, done_mask, dw_mask

    def gae(self, unification=True):
        transitions = [[] for i in range(self.cluster)]
        collect_adv = [[] for i in range(self.cluster)]
        for agent in self.data:
            # name is also index

            cluster_index = agent.name % self.cluster

            s1, s2, _, r, s1_prime, s2_prime, _, done_mask, dw_mask = self.make_batch(agent)
            ''' Use TD+GAE+LongTrajectory to compute Advantage and TD target'''
            self.local_critics[cluster_index].eval()
            with torch.no_grad():
                vs = self.local_critics[cluster_index](s1, s2)
                vs_ = self.local_critics[cluster_index](s1_prime, s2_prime)
                '''dw for TD_target and Adv'''
                deltas = r + self.gamma * vs_ * (1 - dw_mask) - vs
                deltas = deltas.cpu().flatten().numpy()
                adv = [0]
                '''done for GAE'''
                for dlt, mask in zip(deltas[::-1], done_mask.cpu().flatten().numpy()[::-1]):
                    advantage = dlt + self.gamma * self.lambd * adv[-1] * (1 - mask)
                    adv.append(advantage)
                adv.reverse()
                adv = copy.deepcopy(adv[0:-1])
                collect_adv[cluster_index] += adv
                td_target = np.array(adv) + np.array(vs.to('cpu').squeeze(1))
            for i, single_transition in enumerate(self.data[agent]):
                transitions[cluster_index].append(single_transition + [[td_target[i]], adv[i]])
        # adv_mean = np.mean(collect_adv, 1)
        # adv_std = np.std(collect_adv, 1)
        adv_mean = [np.mean(a) for a in collect_adv]
        adv_std = [np.std(a) for a in collect_adv]
        # transitions = [tuple(tran[0:-1] + [[(tran[-1] - adv_mean) / (adv_std + 1e-6)]]) for tran in transitions]
        for i in range(self.cluster):
            transitions[i] = [tuple(tran[0:-1] + [[(tran[-1] - adv_mean[i]) / (adv_std[i] + 1e-6)]]) for tran in
                              transitions[i]]
        return transitions

    def put_data(self, agent, transition):
        # this part is the same as non-fed, the sequence of agent is its agent.name
        if agent in self.data:
            self.data[agent].append(transition)
        else:
            self.data[agent] = [transition]

    def save(self, global_step, index=None):
        # global_step is usually interger, but also could be string for some events
        diff = f"_{index}" if index else ''
        if isinstance(global_step, str):
            global_step = global_step
            seq_name = f"{global_step}{diff}"
        else:
            global_step /= 1e6
            seq_name = f"{global_step}m{diff}"
        torch.save(self.average_model_weights(self.local_actors, for_saving=True),
                   f"{self.dir}/ppo_actor_{seq_name}.pth")
        torch.save(self.average_model_weights(self.local_critics, for_saving=True),
                   f"{self.dir}/ppo_critic_{seq_name}.pth")

    def load(self, folder, global_step, partial_fine_tune=True):
        if isinstance(global_step, float) or isinstance(global_step, int):
            global_step = str(global_step / 1000000) + 'm'
        if folder.startswith('/'):
            saved_actor_weights = torch.load(f"{folder}/ppo_actor_{global_step}.pth")
            saved_critic_weights = torch.load(f"{folder}/ppo_critic_{global_step}.pth")
        else:
            saved_actor_weights = torch.load(f"./{folder}/ppo_actor_{global_step}.pth")
            saved_critic_weights = torch.load(f"./{folder}/ppo_critic_{global_step}.pth")
        for local_actor, local_critic in zip(self.local_actors, self.local_critics):
            local_actor.load_state_dict(saved_actor_weights, strict=False)
            local_critic.load_state_dict(saved_critic_weights, strict=False)
            if partial_fine_tune:
                for model in [local_actor, local_critic]:
                    for name, param in model.named_parameters():
                        if name.startswith('intput_merge') and not name.startswith('intput_merge.trans.fc_module'):
                            param.requires_grad = False


    def weights_track(self, global_step):
        pass
        # total_sum = 0.0
        # for param in self.actor.parameters():
        #     total_sum += torch.sum(param)
        # self.writer.add_scalar("weights/actor_sum", total_sum, global_step)
        # total_sum = 0.0
        # for param in self.critic.parameters():
        #     total_sum += torch.sum(param)
        # self.writer.add_scalar("weights/critic_sum", total_sum, global_step)
