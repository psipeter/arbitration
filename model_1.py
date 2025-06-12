import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys

class Environment():
    def __init__(self, sid, seed=1, t_cue=0.5, t_reward=0.5, dt=0.001, p_reward=1.0,
                 alpha_plus=0.5, alpha_minus=0.5, alpha_unchosen=1.0, omega_0=0.5, alpha_omega=0.3, gamma_omega=0.1):
        self.empirical = pd.read_pickle("data/empirical.pkl")
        self.sid = sid
        self.rng = np.random.RandomState(seed=seed)
        self.t_cue = t_cue
        self.t_reward = t_reward
        self.dt = dt
        self.p_reward = p_reward
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.alpha_unchosen = alpha_unchosen
        self.wab0 = omega_0
        self.wlr0 = 1-omega_0
        self.alpha_omega = alpha_omega
        self.gamma_omega = gamma_omega
        self.F = np.array([0,0,0,0])  # [A, B, L, R]
        self.G = np.array([0,0])  # [A/B, L/R]
        self.letter = [0]
        self.learning_rates = [self.alpha_plus, self.alpha_minus, self.alpha_unchosen, self.alpha_omega, self.gamma_omega, self.wab0, self.wlr0]
        self.reward = [0]
        self.action = [0]
        self.feedback_phase = 0
        self.cue_phase = 0
    def set_cue(self, bid, trial):
        sid = self.sid
        left = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['left'].to_numpy()[0]
        right = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['right'].to_numpy()[0]
        self.letter  = [1] if left=='A' else [-1]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 1
        self.feedback_phase = 0
    def set_action(self, sim, net):
        self.action = [1] if sim.data[net.p_a][-1][0]>sim.data[net.p_a][-1][1] else [-1]
    def set_reward(self, bid, trial):
        sid = self.sid
        block = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['block'].to_numpy()[0]
        correct = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['correct'].to_numpy()[0]
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
        return self.letter
    def sample_letter2(self, t):
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

def build_network(env, n_neurons=3000, seed_network=0, alpha_pes=3e-5):
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
        in_letter2 = nengo.Node(lambda t: env.sample_letter2(t))
        in_action = nengo.Node(lambda t: env.sample_action(t))
        in_reward = nengo.Node(lambda t: env.sample_reward(t))
        in_update = nengo.Node(lambda t: env.sample_update(t))
        in_decay = nengo.Node(lambda t: env.sample_decay(t))
        in_phase = nengo.Node(lambda t: env.sample_phase(t))
        in_w0 = nengo.Node([env.wab0, env.wlr0])
        
        # ensembles and nodes
        f = nengo.Ensemble(n_neurons, 4)
        g = nengo.Ensemble(n_neurons, 2)
        v = nengo.Ensemble(n_neurons, 4, radius=2)
        w = nengo.Ensemble(n_neurons, 2, radius=2)
        a = nengo.Ensemble(n_neurons, 2, radius=2)
        vlet = nengo.Ensemble(n_neurons, 4, radius=2)
        vwa = nengo.Ensemble(n_neurons, 6, radius=3)
        evc = nengo.Ensemble(n_neurons, 8, radius=4)
        evu = nengo.Ensemble(n_neurons, 8, radius=4)
        drel = nengo.Ensemble(n_neurons, 8, radius=4)
        wtar = nengo.Ensemble(n_neurons, 1)
        ewt = nengo.Ensemble(n_neurons, 2, radius=2)
        ewd = nengo.Ensemble(n_neurons, 2, radius=2)
        evcout = nengo.Ensemble(1, 4, neuron_type=nengo.Direct())
        evuout = nengo.Ensemble(1, 4, neuron_type=nengo.Direct())
        drelout = nengo.Ensemble(1, 1, neuron_type=nengo.Direct())
        wtarout = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())

        # connections
        nengo.Connection(in_f, f)
        nengo.Connection(in_g, g)

        cf = nengo.Connection(f, v, synapse=0.01, transform=0, learning_rule_type=pes)
        cg = nengo.Connection(g, w, synapse=0.01, function=lambda x: [env.wab0, env.wlr0], learning_rule_type=pes)

        nengo.Connection(v[:2], vlet[:2], synapse=0.01)
        nengo.Connection(in_letter2, vlet[2:4])
        nengo.Connection(vlet, vwa[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])  # vletl: va if letter==1 else vb
        nengo.Connection(vlet, vwa[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])  # vletr: vb if letter==1 else va
        nengo.Connection(v[2:4], vwa[2:4], synapse=0.01)  # [vl, vr]
        nengo.Connection(w, vwa[4:6], synapse=0.01)  # [wab, wlr]
        nengo.Connection(vwa, a[0], synapse=0.01, function=lambda x: x[0]*x[4]+x[2]*x[5])  # vletl*wab + vl*wlr
        nengo.Connection(vwa, a[1], synapse=0.01, function=lambda x: x[1]*x[4]+x[3]*x[5])  # vletr*wab + vr*wlr

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
        nengo.Connection(wtar, ewt, synapse=0.01, function=lambda x: [1,0] if x>0 else [0,1])
        nengo.Connection(w, ewt, synapse=0.01, transform=-1)
        nengo.Connection(in_phase[0], ewt.neurons, transform=winh)

        nengo.Connection(in_w0, ewd)
        nengo.Connection(w, ewd, synapse=0.01, transform=-1)
        nengo.Connection(in_phase[0], ewd.neurons, transform=winh)

        nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=env.alpha_plus, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-env.alpha_unchosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(ewt, cg.learning_rule, synapse=0.01, transform=-env.alpha_omega)
        nengo.Connection(ewd, cg.learning_rule, synapse=0.01, transform=-env.gamma_omega)

        nengo.Connection(evu, evuout, synapse=0.01, transform=-env.alpha_unchosen, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(evc, evcout, synapse=0.01, transform=env.alpha_plus, function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])
        nengo.Connection(drel, drelout, synapse=0.01, function=lambda x: [x[0]*x[4]+x[1]*x[5]-x[2]*x[6]-x[3]*x[7]])
        nengo.Connection(wtar, wtarout, synapse=0.01, function=lambda x: [1,0] if x>0 else [0,1])
    
        # probes
        net.p_letter = nengo.Probe(in_letter)
        net.p_reward = nengo.Probe(in_reward)
        net.p_v = nengo.Probe(v)
        net.p_w = nengo.Probe(w)
        net.p_a = nengo.Probe(a)
        net.p_vlet = nengo.Probe(vlet)
        net.p_vwa = nengo.Probe(vwa)
        net.p_evc = nengo.Probe(evc)
        net.p_evu = nengo.Probe(evu)
        net.p_ewt = nengo.Probe(ewt)
        net.p_ewd = nengo.Probe(ewd)
        net.p_drel = nengo.Probe(drel)
        net.p_evcout = nengo.Probe(evcout)
        net.p_evuout = nengo.Probe(evuout)
        net.p_drelout = nengo.Probe(drelout)
        net.p_wtarout = nengo.Probe(wtarout)


    return net

def simulate_network(net, blocks=24):
    dfs = []
    columns = ['sid', 'bid', 'trial_before_reversal', 'trial_after_reversal', 'accuracy']
    env = net.env
    sid = env.sid
    sim = nengo.Simulator(net, dt=env.dt, progress_bar=False)
    with sim:
        for bid in env.empirical.query("sid==@sid")['bid'].unique()[:blocks]:
            for trial in env.empirical.query("sid==@sid & bid==@bid")['trial'].unique():
                print(f"running sid {env.sid}, block {bid}, trial {trial}")
                env.set_cue(bid, trial)
                sim.run(env.t_cue)
                env.set_action(sim, net)
                env.set_reward(bid, trial)
                sim.run(env.t_reward)
                block = env.empirical.query("sid==@sid & bid==@bid & trial==@trial")['block'].to_numpy()[0]
                correct = env.empirical.query("sid==@sid & bid==@bid & trial==@trial")['correct'].to_numpy()[0]
                accuracy = 1 if (env.action==[1] and correct=='left') or (env.action==[-1] and correct=='right') else 0
                reversal_at_trial = env.empirical.query("sid==@sid & bid==@bid")['reversal_at_trial'].unique()[0]
                trial_before_reversal = trial if trial<reversal_at_trial else None
                trial_after_reversal = trial - reversal_at_trial if trial>=reversal_at_trial else None
                dfs.append(pd.DataFrame([[sid, bid, trial_before_reversal, trial_after_reversal, accuracy]], columns=columns))
    data = pd.concat(dfs, ignore_index=True)
    return sim, data

if __name__ == "__main__":
    sid = int(sys.argv[1])
    env = Environment(sid=sid)
    net = build_network(env, seed_network=sid)
    sim, data = simulate_network(net)
    data.to_pickle(f"data/model1_sid{sid}_behavior.pkl")
    print(data)