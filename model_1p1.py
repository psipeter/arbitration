import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys
import gc

class Environment():
    def __init__(self, monkey, session, seed=1, t_cue=0.5, t_reward=0.5, dt=0.001, p_reward=0.7,
                 alpha_chosen=0.5, alpha_unchosen=1.0, omega_0=0.5, alpha_omega=0.3, gamma_omega=0.1):
        self.empirical = pd.read_pickle("data/empirical.pkl")
        self.monkey = monkey
        self.session = session
        self.rng = np.random.RandomState(seed=seed)
        self.t_cue = t_cue
        self.t_reward = t_reward
        self.dt = dt
        self.p_reward = p_reward
        self.alpha_chosen = alpha_chosen
        self.alpha_unchosen = alpha_unchosen
        self.w0 = omega_0
        self.alpha_omega = alpha_omega
        self.gamma_omega = gamma_omega
        self.F = np.array([0,0,0,0])  # [A, B, L, R]
        self.G = np.array([0,0])  # [A/B, L/R]
        self.letter = [0]
        self.reward = [0]
        self.action = [0]
        self.feedback_phase = 0
        self.cue_phase = 0
    def set_cue(self, block, trial):
        monkey = self.monkey
        session = self.session
        left = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['left'].to_numpy()[0]
        right = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['right'].to_numpy()[0]
        self.letter  = [1] if left=='A' else [-1]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 1
        self.feedback_phase = 0
    def set_action(self, sim, net):
        self.action = [1] if sim.data[net.p_a][-1][0]>sim.data[net.p_a][-1][1] else [-1]
    def set_reward(self, block, trial):
        monkey = self.monkey
        session = self.session
        block = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['block'].to_numpy()[0]
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
        self.cue_phase = 0
        self.feedback_phase = 1
    def sample_letter(self, t):
        return [1,0] if self.letter==[1] else [0,1]
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

def build_network(env, n_neurons=3000, seed_network=0, alpha_pes=3e-5, run_to_fit=False):
    net = nengo.Network(seed=seed_network)
    net.env = env
    net.config[nengo.Connection].synapse = None
    net.config[nengo.Probe].synapse = 0.1
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
        in_w0 = nengo.Node(env.w0)
        
        # ensembles and nodes
        f = nengo.Ensemble(n_neurons, 4)
        g = nengo.Ensemble(n_neurons, 2)
        v = nengo.Ensemble(n_neurons, 4, radius=2)
        w = nengo.Ensemble(n_neurons, 1)
        a = nengo.Ensemble(n_neurons, 2, radius=2)
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

        cf = nengo.Connection(f, v, synapse=0.01, transform=0, learning_rule_type=pes)
        cg = nengo.Connection(g, w, synapse=0.01, function=lambda x: env.w0, learning_rule_type=pes)

        nengo.Connection(v[:2], vlet[:2], synapse=0.01)
        nengo.Connection(in_letter, vlet[2:4])
        nengo.Connection(vlet, vwa[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])  # vletl: va if letter==1 else vb
        nengo.Connection(vlet, vwa[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])  # vletr: vb if letter==1 else va
        nengo.Connection(v[2:4], vwa[2:4], synapse=0.01)  # [vl, vr]
        nengo.Connection(w, vwa[4], synapse=0.01)
        nengo.Connection(vwa, a[0], synapse=0.01, function=lambda x: x[0]*x[4]+x[2]*(1-x[4]))  # vletl*w + vl*(1-w)
        nengo.Connection(vwa, a[1], synapse=0.01, function=lambda x: x[1]*x[4]+x[3]*(1-x[4]))  # vletr*w + vr*(1-w)

        nengo.Connection(v, evc[:4], synapse=0.01)
        nengo.Connection(in_reward, evc[:4], transform=[[-1],[-1],[-1],[-1]])
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

        nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=env.alpha_chosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-env.alpha_unchosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(ewt, cg.learning_rule, synapse=0.01, transform=-env.alpha_omega)
        nengo.Connection(ewd, cg.learning_rule, synapse=0.01, transform=-env.gamma_omega)

        nengo.Connection(vlet, vletout[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])
        nengo.Connection(vlet, vletout[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])
        nengo.Connection(evu, evuout, synapse=0.01, transform=-env.alpha_unchosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evc, evcout, synapse=0.01, transform=env.alpha_chosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(drel, drelout, synapse=0.01, function=lambda x: [x[0]*x[4]+x[1]*x[5]-x[2]*x[6]-x[3]*x[7]])
        nengo.Connection(wtar, wtarout, synapse=0.01, function=lambda x: [1,0] if x>0 else [0,1])
    
        # probes
        if run_to_fit:
            net.p_a = nengo.Probe(a)
        else:
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
            # net.p_evcout = nengo.Probe(evcout)
            # net.p_evuout = nengo.Probe(evuout)
            # net.p_drelout = nengo.Probe(drelout)
            # net.p_wtarout = nengo.Probe(wtarout)
            net.s_vwa = nengo.Probe(vwa.neurons, synapse=None)
            net.s_evc = nengo.Probe(evc.neurons, synapse=None)
            net.s_a = nengo.Probe(a.neurons, synapse=None)

    return net


def simulate_values_spikes(net, block):
    dfs = []
    columns = ['monkey', 'session', 'block', 'trial', 'block_type', 'before', 'after', 'va', 'vb', 'vl', 'vr', 'w', 'al', 'ar', 'clet', 'cloc', 'rew', 'acc']
    env = net.env
    monkey = env.monkey
    session = env.session
    block_type = 'what' if block<= 12 else 'where'
    sim = nengo.Simulator(net, dt=env.dt, progress_bar=False)
    labels = ['value', 'omega', 'action', 'mixed', 'error', 'reliability']
    with sim:
        for trial in env.empirical.query("monkey==@monkey & session==@session & block==@block")['trial'].unique():
            print(f"running monkey {env.monkey}, session {session}, block {block}, trial {trial}")
            net.env.set_cue(block, trial)
            sim.run(net.env.t_cue)
            t_choice = sim.trange().shape[0]
            t0 = t_choice - 100  # 100ms prior to choice
            correct = env.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['correct'].to_numpy()[0]
            va = sim.data[net.p_v][t0:t_choice,0].mean() if env.letter==[1] else sim.data[net.p_v][t0:t_choice,1].mean()
            vb = sim.data[net.p_v][t0:t_choice,1].mean() if env.letter==[1] else sim.data[net.p_v][t0:t_choice,0].mean()
            vl = sim.data[net.p_v][t0:t_choice,2].mean()
            vr = sim.data[net.p_v][t0:t_choice,3].mean()
            vletl = sim.data[net.p_vlet][t0:t_choice,0].mean()
            vletr = sim.data[net.p_vlet][t0:t_choice,1].mean()
            w = sim.data[net.p_w][t0:t_choice,0].mean()
            al = sim.data[net.p_a][t0:t_choice,0].mean()
            ar = sim.data[net.p_a][t0:t_choice,1].mean()
            svwa = sim.data[net.s_vwa][t0:t_choice].sum(axis=0) / 1000
            env.set_action(sim, net)
            env.set_reward(block, trial)
            clet = 0 if (env.action==[1] and env.letter==[1]) or (env.action==[-1] and env.letter==[-1]) else 1
            cloc = 0 if env.action==[1] else 1
            acc = 1 if (cloc==0 and correct=='left') or (cloc==1 and correct=='right') else 0
            rew = 1 if env.reward[0]==1 else 0
            sim.run(net.env.t_reward)
            t1 = t_choice + 100  # 100ms following choice / reward delivery
            sevc = sim.data[net.s_evc][t_choice:t1].sum(axis=0) / 1000
            sa = sim.data[net.s_a][t_choice:t1].sum(axis=0) / 1000
            reversal_at_trial = env.empirical.query("monkey==@monkey & session==@session & block==@block")['reversal_at_trial'].unique()[0]
            before = trial if trial<reversal_at_trial else None
            after = trial - reversal_at_trial if trial>=reversal_at_trial else None
            df = pd.DataFrame([[monkey, session, block, trial, block_type, before, after,
                va, vb, vl, vr, w, al, ar, clet, cloc, rew, acc]], columns=columns)
            filename = f"data/nef_spikes/monkey{monkey}_session{session}_block{block}_trial{trial}"
            df.to_pickle(filename+"_values.pkl")
            np.savez_compressed(filename+"_spikes.npz", vwa=svwa, evc=sevc, a=sa)

def run_to_fit(monkey, session, alpha_chosen, alpha_unchosen, omega_0, alpha_omega, gamma_omega, neurons):
    dfs = []
    columns = ['monkey', 'session', 'block', 'trial', 'choice']
    seed_network = session + 4 if monkey=='W' else session
    env = Environment(monkey=monkey, session=session, alpha_chosen=alpha_chosen,
        alpha_unchosen=alpha_unchosen, omega_0=omega_0, alpha_omega=alpha_omega,
        gamma_omega=gamma_omega)
    net = build_network(env, n_neurons=neurons, seed_network=seed_network, run_to_fit=True)
    sim = nengo.Simulator(net, dt=env.dt, progress_bar=False)
    with sim:
        for block in env.empirical.query("monkey==@monkey & session==@session")['block'].unique():
            for trial in env.empirical.query("monkey==@monkey & session==@session & block==@block")['trial'].unique():
                print(f"running monkey {env.monkey}, session {session}, block {block}, trial {trial}")
                net.env.set_cue(block, trial)
                sim.run(net.env.t_cue)
                env.set_action(sim, net)
                env.set_reward(block, trial)
                choice = env.action[0]
                sim.run(net.env.t_reward)
                dfs.append(pd.DataFrame([[monkey, session, block, trial, choice]],columns=columns))
    data = pd.concat(dfs, ignore_index=True)
    return data

if __name__ == "__main__":
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    param_config = sys.argv[3]
    seed_network = session + 4 if monkey=='W' else session
    if param_config=='load':
        params = pd.read_pickle(f"data/{monkey}_{session}_params.pkl")
        alpha_chosen = params['alpha_chosen'].unique()[0]
        alpha_unchosen = params['alpha_unchosen'].unique()[0]
        omega_0 = params['omega_0'].unique()[0]
        alpha_omega = params['alpha_omega'].unique()[0]
        gamma_omega = params['gamma_omega'].unique()[0]
        neurons = params['neurons'].unique()[0]
        env = Environment(monkey=monkey, session=session, alpha_chosen=alpha_chosen,
            alpha_unchosen=alpha_unchosen, omega_0=omega_0, alpha_omega=alpha_omega,
            gamma_omega=gamma_omega)
        net = build_network(env, n_neurons=neurons, seed_network=seed_network)
    if param_config=='random':
        rng = np.random.RandomState(seed=seed_network)
        alpha_chosen = rng.uniform(0.4, 0.6)
        alpha_unchosen = rng.uniform(0.9, 1.0)
        omega_0 = rng.uniform(0.4, 0.6)
        alpha_omega = rng.uniform(0.2, 0.4)
        gamma_omega = rng.uniform(0.05, 0.15)
        neurons = 3000
        env = Environment(monkey=monkey, session=session,
            alpha_chosen=alpha_chosen, alpha_unchosen=alpha_unchosen, omega_0=omega_0, alpha_omega=alpha_omega, gamma_omega=gamma_omega)
        net = build_network(env, n_neurons=neurons, seed_network=seed_network)
    elif param_config=='default':
        env = Environment(monkey=monkey, session=session)
        net = build_network(env, seed_network=seed_network)
    else:
        print("Must specify which parameters to use")
        raise

    blocks = 2
    for block in range(1, blocks+1):
        if block in env.empirical.query("monkey==@monkey & session==@session")['block'].unique():
            simulate_values_spikes(net, block)
        else:
            print(f"monkey {monkey} session {session} missing block {block}")