import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys

class Environment():
    def __init__(self, sid, seed=1, t_load=1, t_cue=0.5, t_reward=0.5, dt=0.001, p_reward=1.0,
                 alpha_plus=0.5, alpha_minus=0.5, alpha_unchosen=0.3, omega_0=0.5, alpha_omega=0.3, gamma_omega=0.1):
        self.empirical = pd.read_pickle("data/empirical.pkl")
        self.sid = sid
        self.rng = np.random.RandomState(seed=seed)
        self.t_load = t_load
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
        self.load_phase = 1
    def set_cue(self, bid, trial):
        sid = self.sid
        left = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['left'].to_numpy()[0]
        right = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['right'].to_numpy()[0]
        self.letter  = [1] if left=='A' else [-1]
        self.reward = [0]
        self.action = [0]
        self.cue_phase = 1
        self.feedback_phase = 0
        self.load_phase = 0
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
        self.load_phase = 0
    def sample_letter(self, t):
        return self.letter
    def sample_letter2(self, t):
        return [1,0] if self.letter==[1] else [0,1]
    def sample_action(self, t):
        return self.action
    def sample_update(self, t):
        update_L = 1 if self.action==[1] else 0
        update_R = 1 if self.action==[-1] else 0
        return [update_L, update_R]
    def sample_decay(self, t):
        decay_L = 1 if self.action==[-1] else 0
        decay_R = 1 if self.action==[1] else 0
        return [decay_L, decay_R]
    def sample_match(self, t):
        match = 1 if (self.action==[1] and self.letter==[1]) or (self.action==[-1] and self.letter==[-1]) else 0
        unmatch = 1 if (self.action==[1] and self.letter==[-1]) or (self.action==[-1] and self.letter==[1]) else 0
        return [match, unmatch]
    def sample_reward(self, t):
        return self.reward
    def sample_phase(self, t):
        return [self.cue_phase, self.feedback_phase, self.load_phase]

class EVC(nengo.Node):
    def __init__(self, learning_rates):
        self.learning_rates = learning_rates
        self.size_in = 10
        self.size_out = 4
        super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
    def step(self, t, x):
        va = x[0]
        vb = x[1]
        vl = x[2]
        vr = x[3]
        reward = x[4]
        update_L = x[5]
        update_R = x[6]
        match = x[7]
        unmatch = x[8]
        phase = x[9]
        alpha_plus = self.learning_rates[0]
        alpha_minus = self.learning_rates[1]
        alpha_unchosen = self.learning_rates[2]
        alpha = alpha_plus if reward==1 else alpha_minus
        dv = [0,0,0,0]  # [va, vb, vl, vr]
        if phase==1:
            dv[0] += -match * alpha*(reward-va)
            dv[1] += -unmatch * alpha*(reward-vb)
            dv[2] += -update_L * alpha*(reward-vl)
            dv[3] += -update_R * alpha*(reward-vr)
        return dv

class EVU(nengo.Node):
    def __init__(self, learning_rates):
        self.learning_rates = learning_rates
        self.size_in = 9
        self.size_out = 4
        super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
    def step(self, t, x):
        va = x[0]
        vb = x[1]
        vl = x[2]
        vr = x[3]
        decay_L = x[4]
        decay_R = x[5]
        match = x[6]
        unmatch = x[7]
        phase = x[8]
        alpha_unchosen = self.learning_rates[2]
        dv = [0,0,0,0]  # [va, vb, vl, vr]
        if phase==1:
            dv[0] += unmatch * (1-alpha_unchosen)*va
            dv[1] += match * (1-alpha_unchosen)*vb
            dv[2] += decay_L * (1-alpha_unchosen)*vl
            dv[3] += decay_R * (1-alpha_unchosen)*vr
        return dv

class EW(nengo.Node):
    def __init__(self, learning_rates):
        self.learning_rates = learning_rates
        self.size_in = 11
        self.size_out = 2
        super().__init__(self.step, size_in=self.size_in, size_out=self.size_out)
    def step(self, t, x):
        va = x[0]
        vb = x[1]
        vl = x[2]
        vr = x[3]
        wab = x[4]
        wlr = x[5]
        reward = x[6]
        action = x[7]
        letter = x[8]
        phase = x[9]
        load = x[10]
        alpha_omega = self.learning_rates[3]
        gamma_omega = self.learning_rates[4]
        wab0 = self.learning_rates[5]
        wlr0 = self.learning_rates[6]
        dw = [0,0]  # [wab, wlr]
        if action==1:
            vca = vl
            if letter==1:
                vcs = va
            elif letter==-1:
                vcs = vb
        elif action==-1:
            vca = vr
            if letter==1:
                vcs = vb
            elif letter==-1:
                vcs = va
        else:
            vca = 0
            vcs = 0
        delta_rel = vcs - vca
        if delta_rel>0:
            tab = 1
            tlr = 0
        elif delta_rel<0:
            tab = 0
            tlr = 1
        else:
            tab = None
            tlr = None
        if phase==1:
            dw[0] += -alpha_omega*np.abs(delta_rel)*(tab-wab)
            dw[1] += -alpha_omega*np.abs(delta_rel)*(tlr-wlr)
            dw[0] += -gamma_omega*(wab0 - wab)
            dw[1] += -gamma_omega*(wlr0 - wlr)
        if load==1:  # speed up initial loading
            dw[0] += -3*(wab0 - wab)
            dw[1] += -3*(wlr0 - wlr)
        return dw

def build_network(env, n_neurons=2000, seed_network=1, inh=0, k=1.0, alpha_pes=1e-4):
    net = nengo.Network(seed=seed_network)
    net.env = env
    net.config[nengo.Connection].synapse = None
    net.config[nengo.Probe].synapse = 0.1
    inh_feedback = -1000*np.ones((n_neurons, 1))
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
        in_match = nengo.Node(lambda t: env.sample_match(t))
        in_phase = nengo.Node(lambda t: env.sample_phase(t))
        
        # ensembles and nodes
        f = nengo.Ensemble(n_neurons, 4)
        g = nengo.Ensemble(n_neurons, 2)
        v = nengo.Ensemble(n_neurons, 4, radius=2)
        w = nengo.Ensemble(n_neurons, 2, radius=2)
        a = nengo.Ensemble(n_neurons, 2, radius=2)
        vlet = nengo.Ensemble(n_neurons, 4, radius=2)
        vwa = nengo.Ensemble(n_neurons, 6, radius=3)
        evc = EVC(env.learning_rates)
        evu = EVU(env.learning_rates)
        ew = EW(env.learning_rates)

        # connections
        nengo.Connection(in_f, f)
        nengo.Connection(in_g, g)

        cf = nengo.Connection(f, v, synapse=0.01, transform=0, learning_rule_type=pes)
        cg = nengo.Connection(g, w, synapse=0.01, transform=0, learning_rule_type=pes)

        nengo.Connection(v[:2], vlet[:2], synapse=0.01)
        nengo.Connection(in_letter2, vlet[2:4])
        nengo.Connection(vlet, vwa[0], synapse=0.01, function=lambda x: x[0]*x[2]+x[1]*x[3])  # vletl: va if letter==1 else vb
        nengo.Connection(vlet, vwa[1], synapse=0.01, function=lambda x: x[1]*x[2]+x[0]*x[3])  # vletr: vb if letter==1 else va
        nengo.Connection(v[2:4], vwa[2:4], synapse=0.01)  # [vl, vr]
        nengo.Connection(w, vwa[4:6], synapse=0.01)  # [wab, wlr]
        nengo.Connection(vwa, a[0], synapse=0.01, function=lambda x: x[0]*x[4]+x[2]*x[5])  # vletl*wab + vl*wlr
        nengo.Connection(vwa, a[1], synapse=0.01, function=lambda x: x[1]*x[4]+x[3]*x[5])  # vletr*wab + vr*wlr

        nengo.Connection(v, evc[:4], synapse=0.01)
        nengo.Connection(in_reward, evc[4])
        nengo.Connection(in_update[0], evc[5])
        nengo.Connection(in_update[1], evc[6])
        nengo.Connection(in_match[0], evc[7])
        nengo.Connection(in_match[1], evc[8])
        nengo.Connection(in_phase[1], evc[9])  # feedback

        nengo.Connection(v, evu[:4], synapse=0.01)
        nengo.Connection(in_decay[0], evu[4])
        nengo.Connection(in_decay[1], evu[5])
        nengo.Connection(in_match[0], evu[6])
        nengo.Connection(in_match[1], evu[7])
        nengo.Connection(in_phase[1], evu[8])  # feedback

        nengo.Connection(v, ew[:4], synapse=0.01)
        nengo.Connection(w, ew[4:6], synapse=0.01)
        nengo.Connection(in_reward, ew[6])
        nengo.Connection(in_action, ew[7])
        nengo.Connection(in_letter, ew[8])
        nengo.Connection(in_phase[1], ew[9])  # feedback
        nengo.Connection(in_phase[2], ew[10])  # load

        nengo.Connection(evc, cf.learning_rule)
        nengo.Connection(evu, cf.learning_rule)
        nengo.Connection(ew, cg.learning_rule)
    
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
        net.p_ew = nengo.Probe(ew)


    return net

def simulate_network(net, blocks=24):
    dfs = []
    columns = ['sid', 'bid', 'trial_before_reversal', 'trial_after_reversal', 'accuracy']
    env = net.env
    sid = env.sid
    sim = nengo.Simulator(net, dt=env.dt, progress_bar=False)
    with sim:
        sim.run(net.env.t_load)
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