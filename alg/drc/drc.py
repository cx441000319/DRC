import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
import spinup.algos.pytorch.drc.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

import copy
import math
import re
import random
import os

class DRCBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)

        self.rew_buf = np.zeros(size, dtype=np.float32)     # estimated reward for computation of advantages
        self.obs_rew_buf = np.zeros(size, dtype=np.float32) # observed reward as the target for reward prediction

        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, est_rew, obs_rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        
        self.rew_buf[self.ptr] = est_rew
        self.obs_rew_buf[self.ptr] = obs_rew

        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    # def finish_path(self, last_val=0):
    #     """
    #     Call this at the end of a trajectory, or when one gets cut off
    #     by an epoch ending. This looks back in the buffer to where the
    #     trajectory started, and uses rewards and value estimates from
    #     the whole trajectory to compute advantage estimates with GAE-Lambda,
    #     as well as compute the rewards-to-go for each state, to use as
    #     the targets for the value function.

    #     The "last_val" argument should be 0 if the trajectory ended
    #     because the agent reached a terminal state (died), and otherwise
    #     should be V(s_T), the value function estimated for the last state.
    #     This allows us to bootstrap the reward-to-go calculation to account
    #     for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
    #     """

    #     path_slice = slice(self.path_start_idx, self.ptr)
    #     rews = np.append(self.rew_buf[path_slice], last_val)
    #     vals = np.append(self.val_buf[path_slice], last_val)
        
    #     # the next two lines implement GAE-Lambda advantage calculation
    #     deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    #     self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
    #     # the next line computes rewards-to-go, to be targets for the value function
    #     self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
    #     self.path_start_idx = self.ptr

    def finish_path_for_drc(self, last_val, start_idx, end_idx):

        path_slice = slice(start_idx, end_idx)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf, rew = self.rew_buf, obs_rew = self.obs_rew_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in data.items()}

def drc(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=250, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        gcm_mode = 0, prob = .0, noise_disc = 10, cont_mode = 0, sigma=0.0,
        num_outputs = 0,):
    
    # gcm_mode: 0 - no GCM perturbations; 1 - GCM perturbations
    # prob: the probability of the samples to be perturbed of the GCM perturbations
    # noise_disc: the reward discretization number of the GCM perturbations
    # cont_mode: 0 - no continuous perturbations; 1 - Gaussian perturbations;
    #            2 - uniform perturbations; 3 - reward range uniform perturbations
    # sigma: sigma/omega of the continuous perturbations
    # num_outputs: the number of outputs from drc

    """
    Proximal Policy Optimization (by clipping), 

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a 
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v`` 
            module. The ``step`` method should accept a batch of observations 
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of 
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing 
                                           | the log probability, according to 
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical: 
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`. 

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()

    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Instantiate environment
    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    # Set up experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = DRCBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # exp_env
    if re.search('Hopper', logger_kwargs['exp_name']):
        known_r_min = 0.5
        known_r_max = 6.5
    elif re.search('HalfCheetah', logger_kwargs['exp_name']):
        known_r_min = -5.
        known_r_max = 7.
    elif re.search('Walker2d', logger_kwargs['exp_name']):
        known_r_min = -3.
        known_r_max = 8.
    elif re.search('Reacher', logger_kwargs['exp_name']):
        known_r_min = -12.
        known_r_max = 0.
    
    # drc definition
    def build_reward_critic(num_inputs, num_outputs, hidden_size):
        return nn.ModuleList([nn.Linear(num_inputs, hidden_size), nn.Linear(hidden_size, hidden_size), nn.Linear(hidden_size, num_outputs)])

    def format_r_input(s, s_prime):
        return torch.cat([s, s_prime], dim=-1)
    
    def rp_forward(rp, inputs):
        x = inputs
        for i, l in enumerate(rp):
            x = l(x)
            if i != len(rp) - 1:
                x = torch.tanh(x)
            else:
                x = torch.softmax(x, dim=1)
        return x

    def predict_reward(rp, s, s_prime):
        r_hat_input = format_r_input(s, s_prime)
        return rp_forward(rp, r_hat_input)
    
    def indices_to_one_hot(indices, num_classes):
        one_hot = torch.zeros((indices.shape[0], num_classes))
        one_hot[torch.arange(indices.shape[0]), indices] = 1
        return one_hot
    
    rp = build_reward_critic(obs_dim[0] * 2, num_outputs=num_outputs, hidden_size=64)
    rp_optimizer = Adam(rp.parameters(), lr=1e-3)

    # compute the distributional reward critic (drc) loss
    def compute_loss_r(rp, obs, obs_prime, obs_rew):

        # compute the observed reward labels
        l_r = (known_r_max - known_r_min) / num_outputs
        obs_rew_cat = torch.floor((obs_rew - known_r_min) // l_r)
        obs_rew_cat = obs_rew_cat.type(torch.int64)
        
        # for incident labels
        obs_rew_cat = torch.where(obs_rew_cat > num_outputs - 1, torch.tensor([num_outputs - 1]), obs_rew_cat)
        obs_rew_cat = torch.where(obs_rew_cat < 0, torch.tensor([0]), obs_rew_cat)

        # compute the OCE loss
        rs_hat = torch.squeeze(predict_reward(rp, obs, obs_prime))
        loss = torch.sum((1.+torch.abs(torch.argmax(rs_hat, dim=1)-obs_rew_cat)/(num_outputs-1.)) * (-torch.sum(indices_to_one_hot(obs_rew_cat, num_outputs) * torch.log(rs_hat), dim=1))/steps_per_epoch)
        return loss
    
    # Set up function for computing PPO policy loss
    def compute_loss_pi(data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret)**2).mean()

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data = buf.get()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            kl = mpi_avg(pi_info['kl'])
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            mpi_avg_grads(ac.pi)    # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            mpi_avg_grads(ac.v)    # average grads across MPI processes
            vf_optimizer.step()

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    # Prepare for interaction with environment
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # GCM confusion matrix generation
    if gcm_mode != 0:
        numberList = []
        for i in range(noise_disc):
            numberList.append(i)
        alphas = []
        for i in range(noise_disc):
            x = i + 1
            p = np.random.rand(noise_disc-1)
            sum_p = sum(p)
            p = p/sum_p * prob
            p = np.insert(p, [i], [1-prob])
            alphas.append(p)
        alphas = np.array(alphas, dtype=float)
    
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        
        # for drc
        ep_lens = []
        last_vals = []
        ep_last_obss = []

        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # GCM peturbations
            if gcm_mode != 0:
                # clean reward label
                noise_interval = (known_r_max - known_r_min)/noise_disc
                x = math.floor((r - known_r_min)/noise_interval)
                
                cp_x = x
                if cp_x < 0:
                    cp_x = 0
                elif cp_x > noise_disc-1:
                    cp_x = noise_disc-1
                probs = alphas[cp_x]

                for i in range(len(probs)):
                    assert(probs[i] >= 0)
                
                # noisy reward label
                mod_x = random.choices(numberList, weights=probs, k=1)[0]

                # according to the difference between the sampled number and x, give a noisy reward
                r = r + (mod_x - x) * noise_interval
            # continuous peturbations
            elif cont_mode != 0:
                if cont_mode == 1:
                    r += np.random.normal(0, sigma)
                elif cont_mode == 2:
                    if np.random.uniform() < sigma:
                        r = np.random.uniform(-1,1)
                elif cont_mode == 3:
                    if np.random.uniform() < sigma:
                        r = np.random.uniform(known_r_min, known_r_max)

            # save and log
            buf.store(o, a, r, r, v, logp) # store the observed rewards as the estimated ones for now
            logger.store(VVals=v)
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                
                # for drc
                last_vals.append(v)

                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                
                # for drc
                ep_lens.append(ep_len)
                ep_last_obss.append(o)

                o, ep_ret, ep_len = env.reset(), 0, 0


        # ---------------------------------------
        # drc training
        rew_NN_ite = 40
        obs, act, obs_rew = buf.obs_buf, buf.act_buf, buf.obs_rew_buf
        obs = torch.from_numpy(obs)
        act = torch.from_numpy(act)
        obs_rew = torch.from_numpy(obs_rew)

        # next_obss: they are the next observations (s_1~s_{t+1}) w.r.t. to the current observations (s_0~s_t)
        pos=1
        for i in range(len(ep_last_obss)):
            if pos==1:
                next_obss=obs[pos:pos+ep_lens[i]-1]
            else:
                next_obss = np.concatenate((next_obss, obs[pos:pos+ep_lens[i]-1]),axis=0)
            next_obss = np.concatenate((next_obss, np.expand_dims(ep_last_obss[i], axis=0)),axis=0)
            pos=pos+ep_lens[i]
        next_obss = torch.from_numpy(copy.deepcopy(next_obss).astype(np.float32))
        
        for i in range(rew_NN_ite):
            rp_optimizer.zero_grad()
            loss_r = compute_loss_r(rp, obs, next_obss, obs_rew)
            loss_r.backward()
            rp_optimizer.step()
        

        # ---------------------------------------
        # drc predicting

        r_pre = predict_reward(rp, obs, next_obss)
        r_pre_cat = torch.argmax(r_pre, dim=1)
        l_r = (known_r_max - known_r_min) / num_outputs
        r_cat = torch.floor((obs_rew - known_r_min) // l_r)
        r_hat = obs_rew + (r_pre_cat - r_cat) * l_r # the estimated rewards
        buf.rew_buf = r_hat.detach().numpy() # store the estimated rewards for the RL update
        
        # # metric: ordinal cross-entropy
        # r_cat_np = r_cat.numpy()
        # rs_hat = torch.squeeze(r_pre.detach())
        # obs_rew_cat = r_cat_np
        # ord_ce = torch.sum((1.+torch.abs(torch.argmax(rs_hat, dim=1)-obs_rew_cat)/(num_outputs-1.)) * (-torch.sum(indices_to_one_hot(obs_rew_cat, num_outputs) * torch.log(rs_hat), dim=1))/steps_per_epoch).item()
        # logger.store(OCE=ord_ce)
        # ---------------------------------------

        # finish the path for drc
        sum_ep_len = 0
        for i in range(len(ep_lens)):
            buf.finish_path_for_drc(last_vals[i], sum_ep_len, sum_ep_len + ep_lens[i])    # compute the advantages and target returns
            sum_ep_len += ep_lens[i]

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        # logger.log_tabular('OCE', average_only=True)
        logger.dump_tabular()

    # metrics: performance
    folder_path = "performance/drc"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    test_episode_rewards = open("performance/drc/" + logger_kwargs['exp_name'] + "_s" + str(seed) + ".txt", "w")
    
    # evaluate for 20 episodes
    for epoch in range(20):

        for t in range(local_steps_per_epoch):
            a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)

            ep_ret += r
            ep_len += 1
            
            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t==local_steps_per_epoch-1

            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                # take notes
                test_episode_rewards.write(str(ep_ret) + "\n")
                o, ep_ret, ep_len = env.reset(), 0, 0
                # critical to jump to the next trajectory!
                break
    
    test_episode_rewards.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='drc')
    parser.add_argument('--gcm_mode', type=int, default=0)
    parser.add_argument('--prob', type=float, default=.0)
    parser.add_argument('--noise_disc', type=int, default=10)
    parser.add_argument('--cont_mode', type=int, default=0)
    parser.add_argument('--sigma', type=float, default=0.0)
    parser.add_argument('--num_outputs', type=int, default=0)
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    drc(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, 
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)