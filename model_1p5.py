import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys
import gc
import time
import pickle
import json
import h5py

class Environment():
    def __init__(self, monkey, session, block, seed_reward, perturb, params,
                t_iti=1.5, t_cue=0.5, t_reward=1.0, dt=0.001, p_reward=0.7):
        self.empirical = pd.read_pickle("data/empirical.pkl")
        self.monkey = monkey
        self.session = session
        self.block = block
        self.perturb = perturb
        self.rng = np.random.default_rng(seed=seed_reward)
        self.t_iti = t_iti
        self.t_cue = t_cue
        self.t_reward = t_reward
        self.dt = dt
        self.p_reward = p_reward
        self.F = np.array([0,0,0,0])  # [A, B, L, R]
        self.G = np.array([0,0])  # [A/B, L/R]
        self.letter = [0]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 0
        self.feedback_phase = 0
        self.params = params
    def set_iti(self):
        self.letter  = [0]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 1  # ITI portion: no letters are presented
        self.feedback_phase = 0
    def set_cue(self, block, trial):
        monkey = self.monkey
        session = self.session
        block = self.block
        self.left = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['left'].to_numpy()[0]
        self.right = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['right'].to_numpy()[0]
        self.letter  = [1] if self.left=='A' else [-1]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 1  # CUE portion: letters are presented
        self.feedback_phase = 0
    def set_action(self, sim, net):
        self.action = [1] if sim.data[net.p_a][-100:][0].mean()>sim.data[net.p_a][-100][1].mean() else [-1]
    def set_reward(self, block, trial):
        monkey = self.monkey
        session = self.session
        block = self.block
        correct = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['correct'].to_numpy()[0]
        deliver_reward = self.rng.uniform(0,1)
        if (self.action == [1] and correct=='left') or (self.action == [-1] and correct=='right'):
            if deliver_reward<=self.p_reward:
                self.reward = [1]  # yes rewarded for picking the better option
            else:
                self.reward = [-1]  # not rewarded for picking the better option
        elif (self.action == [1] and correct=='right') or (self.action == [-1] and correct=='left'):
            if deliver_reward<=(1-self.p_reward):
                self.reward = [1]  # yes rewarded for picking the worse option
            else:
                self.reward = [-1]  # not rewarded for picking the worse option
        else:
            raise
        self.correct = correct
        self.cue_phase = 0
        self.feedback_phase = 1
    def sample_letter(self, t):
        if self.letter==[1]:
            return [1,0]
        elif self.letter==[-1]:
            return [0,1]
        else:
            return [0,0]
    def sample_action(self, t):
        return self.action
    def sample_update(self, t):
        update = [0,0,0,0]
        update[0] = 1 if (self.action==[1] and self.letter==[1]) or (self.action==[-1] and self.letter==[-1]) else 0
        update[1] = 1 if (self.action==[1] and self.letter==[-1]) or (self.action==[-1] and self.letter==[1]) else 0
        update[2] = 1 if self.action==[1] else 0
        update[3] = 1 if self.action==[-1] else 0
        return update
    def sample_decay(self, t):
        decay = [0,0,0,0]
        decay[0] = 1 if (self.action==[1] and self.letter==[-1]) or (self.action==[-1] and self.letter==[1]) else 0
        decay[1] = 1 if (self.action==[1] and self.letter==[1]) or (self.action==[-1] and self.letter==[-1]) else 0
        decay[2] = 1 if self.action==[-1] else 0
        decay[3] = 1 if self.action==[1] else 0
        return decay
    def sample_reward(self, t):
        return self.reward
    def sample_phase(self, t):
        return [self.cue_phase, self.feedback_phase]
    def sample_perturb(self, t):
        return self.perturb if self.cue_phase else 0

def build_network(env, n_neurons=3000, seed_network=0, alpha_pes=3e-5):
    net = nengo.Network(seed=seed_network)
    net.env = env
    net.config[nengo.Connection].synapse = None
    net.config[nengo.Probe].synapse = 0.01
    winh = -1000*np.ones((n_neurons, 1))
    pes = nengo.PES(learning_rate=alpha_pes)

    with net:
        # inputs
        in_f = nengo.Node(env.F)
        in_g = nengo.Node(env.G)
        in_letter = nengo.Node(lambda t: env.sample_letter(t))
        in_action = nengo.Node(lambda t: env.sample_action(t))
        in_reward = nengo.Node(lambda t: env.sample_reward(t))
        in_update = nengo.Node(lambda t: env.sample_update(t))
        in_decay = nengo.Node(lambda t: env.sample_decay(t))
        in_phase = nengo.Node(lambda t: env.sample_phase(t))
        in_w0 = nengo.Node(env.params['w0'])
        in_perturb = nengo.Node(lambda t: env.sample_perturb(t))
        
        # ensembles and nodes
        f = nengo.Ensemble(n_neurons, 4)
        g = nengo.Ensemble(n_neurons, 2)
        v = nengo.Ensemble(n_neurons, 4, radius=2)
        w = nengo.Ensemble(n_neurons, 1)
        a = nengo.Ensemble(n_neurons, 2, radius=2)
        r = nengo.Ensemble(n_neurons, 2, radius=2)
        vlet = nengo.Ensemble(n_neurons, 4, radius=2)
        vwa = nengo.Ensemble(n_neurons, 5, radius=3)
        evc = nengo.Ensemble(n_neurons, 8, radius=4)
        evu = nengo.Ensemble(n_neurons, 8, radius=4)
        drel = nengo.Ensemble(n_neurons, 8, radius=4)
        wtar = nengo.Ensemble(n_neurons, 1)
        ewt = nengo.Ensemble(n_neurons, 1)
        ewd = nengo.Ensemble(n_neurons, 1)
        vletout = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())
        evcout = nengo.Ensemble(1, 4, neuron_type=nengo.Direct())
        evuout = nengo.Ensemble(1, 4, neuron_type=nengo.Direct())
        drelout = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        wtarout = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        # connections
        nengo.Connection(in_f, f)
        nengo.Connection(in_g, g)

        cf = nengo.Connection(f[:2], v[:2], synapse=0.01, transform=0, learning_rule_type=pes)  # learned connection for letters
        cf = nengo.Connection(f[2:], v[2:], synapse=0.01, transform=0, learning_rule_type=pes/2)  # learned connection for locations
        cg = nengo.Connection(g, w, synapse=0.01, function=lambda x: env.params['w0'], learning_rule_type=pes)  # learned connection for omega
        cp = nengo.Connection(in_perturb, w, synapse=None)

        nengo.Connection(v[:2], vlet[:2], synapse=0.01)
        nengo.Connection(in_letter, vlet[2:4])
        nengo.Connection(vlet, vwa[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])  # vletl: va if letter==1 else vb
        nengo.Connection(vlet, vwa[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])  # vletr: vb if letter==1 else va
        nengo.Connection(v[2:4], vwa[2:4], synapse=0.01)  # [vl, vr]
        nengo.Connection(w, vwa[4], synapse=0.01)

        nengo.Connection(vwa, a[0], synapse=0.01, transform=env.params['ff'], function=lambda x: x[0]*x[4]+x[2]*(1-x[4]))  # vletl*w + vl*(1-w)
        nengo.Connection(vwa, a[1], synapse=0.01, transform=env.params['ff'], function=lambda x: x[1]*x[4]+x[3]*(1-x[4]))  # vletr*w + vr*(1-w)
        nengo.Connection(a, a, synapse=0.1)  # a becomes neural integrator
        nengo.Connection(a, r, synapse=0.01, transform=-1)
        nengo.Connection(r, a, synapse=0.1)
        nengo.Connection(in_phase[0], r.neurons, transform=winh)  # reset is inhibited during cue phase

        nengo.Connection(v, evc[:4], synapse=0.01)
        nengo.Connection(in_reward, evc[:4], transform=4*[[-1]])
        nengo.Connection(in_update, evc[4:8])
        nengo.Connection(in_phase[0], evc.neurons, transform=winh)

        nengo.Connection(v, evu[:4], synapse=0.01, transform=-1)
        nengo.Connection(in_decay, evu[4:8])
        nengo.Connection(in_phase[0], evu.neurons, transform=winh)

        nengo.Connection(v, drel[:4], synapse=0.01)
        nengo.Connection(in_update, drel[4:8])
        nengo.Connection(drel, wtar, synapse=0.01, function=lambda x: [x[0]*x[4]+x[1]*x[5]-x[2]*x[6]-x[3]*x[7]])
        nengo.Connection(wtar, ewt, synapse=0.01, function=lambda x: 1 if x>0 else 0)
        nengo.Connection(w, ewt, synapse=0.01, transform=-1)
        nengo.Connection(in_phase[0], ewt.neurons, transform=winh)

        nengo.Connection(in_w0, ewd)
        nengo.Connection(w, ewd, synapse=0.01, transform=-1)
        nengo.Connection(in_phase[0], ewd.neurons, transform=winh)

        nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=env.params['alpha_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-env.params['gamma_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(ewt, cg.learning_rule, synapse=0.01, transform=-env.params['alpha_w'])
        nengo.Connection(ewd, cg.learning_rule, synapse=0.01, transform=-env.params['gamma_w'])

        nengo.Connection(vlet, vletout[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])
        nengo.Connection(vlet, vletout[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])
        nengo.Connection(evc, evcout, synapse=0.01, transform=env.params['alpha_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evu, evuout, synapse=0.01, transform=-env.params['gamma_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(drel, drelout, synapse=0.01, function=lambda x: [x[0]*x[4]+x[1]*x[5]-x[2]*x[6]-x[3]*x[7]])
        nengo.Connection(wtar, wtarout, synapse=0.01, function=lambda x: [1,0] if x>0 else [0,1])
    
        # probes
        net.p_reward = nengo.Probe(in_reward)
        net.p_v = nengo.Probe(v)
        net.p_w = nengo.Probe(w)
        net.p_a = nengo.Probe(a)
        net.p_vlet = nengo.Probe(vletout)
        net.p_vwa = nengo.Probe(vwa)
        net.p_evc = nengo.Probe(evc)
        net.p_evu = nengo.Probe(evu)
        net.p_ewt = nengo.Probe(ewt)
        net.p_ewd = nengo.Probe(ewd)
        net.p_drel = nengo.Probe(drel)
        net.s_vwa = nengo.Probe(vwa.neurons, synapse=None)
        net.s_a = nengo.Probe(a.neurons, synapse=None)

    return net


def simulate_values_spikes(net):
    dfs = []
    # spikes = {'vwa':{}, 'a':{}}
    spikes = {}
    columns = ['monkey', 'session', 'block', 'trial', 'trial_rev', 'block_type', 'perturb',
                'va', 'vb', 'vl', 'vr', 'omega', 'al', 'ar',
                'clet', 'cloc', 'rew', 'acc', 'dvs', 'dva']
    env = net.env
    monkey = env.monkey
    session = env.session
    block = env.block
    perturb = env.perturb
    block_type = 'what' if block<=12 else 'where'
    sim = nengo.Simulator(net, dt=env.dt, progress_bar=False)
    labels = ['value', 'omega', 'action', 'mixed', 'error', 'reliability']
    with sim:
        for trial in env.empirical.query("monkey==@monkey & session==@session & block==@block")['trial'].unique():
            print(f"running monkey {env.monkey}, session {session}, block {block}, trial {trial}, perturb {perturb}")
            t_start = sim.trange().shape[0]
            net.env.set_iti()
            sim.run(net.env.t_iti)
            net.env.set_cue(block, trial)
            sim.run(net.env.t_cue)
            t_choice = sim.trange().shape[0]
            t0 = t_choice - 20  # 20ms prior to choice
            va = sim.data[net.p_v][t0:t_choice,0].mean() if env.letter==[1] else sim.data[net.p_v][t0:t_choice,1].mean()
            vb = sim.data[net.p_v][t0:t_choice,1].mean() if env.letter==[1] else sim.data[net.p_v][t0:t_choice,0].mean()
            vl = sim.data[net.p_v][t0:t_choice,2].mean()
            vr = sim.data[net.p_v][t0:t_choice,3].mean()
            vletl = sim.data[net.p_vlet][t0:t_choice,0].mean()
            vletr = sim.data[net.p_vlet][t0:t_choice,1].mean()
            w = sim.data[net.p_w][t0:t_choice,0].mean()
            al = sim.data[net.p_a][t0:t_choice,0].mean()
            ar = sim.data[net.p_a][t0:t_choice,1].mean()
            env.set_action(sim, net)
            env.set_reward(block, trial)
            clet = 'A' if (env.action==[1] and env.letter==[1]) or (env.action==[-1] and env.letter==[-1]) else 'B'
            cloc = 'left' if env.action==[1] else 'right'
            acc = 1 if (cloc=='left' and env.correct=='left') or (cloc=='right' and env.correct=='right') else 0
            rew = 1 if env.reward[0]==1 else -1
            sim.run(net.env.t_reward)
            t_end = sim.trange().shape[0]
            svwa = sim.data[net.s_vwa][t_start: t_end] / 1000
            sa = sim.data[net.s_a][t_start: t_end] / 1000
            reversal_at_trial = env.empirical.query("monkey==@monkey & session==@session & block==@block")['reversal_at_trial'].unique()[0]
            trial_rev = trial.astype('int64') - reversal_at_trial.astype('int64')
            df = pd.DataFrame([[monkey, session, block, trial, trial_rev, block_type, perturb,
                va, vb, vl, vr, w, al, ar,
                clet, cloc, rew, acc, va-vb, vl-vr]], columns=columns)
            dfs.append(df)
            # spikes['vwa'][trial] = svwa
            # spikes['a'][trial] = sa
            spikes[trial] = sa
    values = pd.concat(dfs, ignore_index=True)
    return values, spikes

def save_spikes_hdf5_binned(spikes, filename, bin_size=100):
    """
    Save NEF spike data (T × N) into HDF5, binning spikes into bin_size-ms windows (same as monkey spike binning).
    spikes[trial] -> ndarray shaped (T, N)
    """
    with h5py.File(filename, "w") as f:
        for trial, arr in spikes.items():
            T, n_neurons = arr.shape
            n_bins = T // bin_size
            T_use = n_bins * bin_size
            arr = arr[:T_use, :]
            arr_binned = arr.reshape(n_bins, bin_size, n_neurons).sum(axis=1)
            f.create_dataset(str(trial), data=arr_binned, compression="gzip", compression_opts=4, shuffle=True)


if __name__ == "__main__":
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    block = int(sys.argv[3])
    param_config = 'load'
    seed_network = session if monkey=='V' else session + 100
    seed_reward = block + 100*session + 1000 if monkey=='V' else block + 100*session + 2000
    if param_config=='load':
        with open("data/rl_fitted_params.json") as f:  # from Jay's simpler RL model fit to the monkey data
            params = json.load(f)[monkey][str(session)]
        params['ff'] = 0.2
        params['alpha_w'] = 2*params['alpha_w']  # try this to boost omega dynamics
    elif param_config=='random':
        rng_network = np.random.default_rng(seed=seed_network)
        params = {
            'alpha_v':rng_network.uniform(0.4, 0.6),
            'gamma_v':rng_network.uniform(0.9, 1.0),
            'w0':rng_network.uniform(0.4, 0.6),
            'alpha_w':rng_network.uniform(0.3, 0.5),
            'gamma_w':rng_network.uniform(0.05, 0.10),
        }
    s = time.time()
    env = Environment(monkey=monkey, session=session, block=block, seed_reward=seed_reward, params=params, perturb=0)
    net = build_network(env, seed_network=seed_network)
    values, spikes = simulate_values_spikes(net)
    filename = f"data/nef/monkey{monkey}_session{session}_block{block}"
    values.to_pickle(filename+"_values.pkl")
    save_spikes_hdf5_binned(spikes, filename + "_spikes.h5")
    e = time.time()
    print(f"runtime (min): {(e-s)/60:.4}")