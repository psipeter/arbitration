import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import scipy
import pandas as pd
import sys

class Environment():
    def __init__(self, sid, seed=1, t_load=1, t_cue=0.5, t_reward=0.5, dt=0.001,
    			 noise_freq=10, noise_rms=0.05, p_reward=1,
                 alpha_plus=0.5, alpha_minus=0.5, gamma_u=0.32,
                 omega_0=0.5, alpha_omega=0.31, gamma_omega=0.09):
        self.empirical = pd.read_pickle("data/empirical.pkl")
        self.sid = sid
        self.rng = np.random.RandomState(seed=seed)
        self.t_load = t_load
        self.t_cue = t_cue
        self.t_reward = t_reward
        self.dt = dt
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.gamma_u = gamma_u
        self.omega_0 = omega_0
        self.alpha_omega = alpha_omega
        self.gamma_omega = gamma_omega
        self.p_reward = p_reward
        self.letter  = [0,0]  # [A=-1, B=1]
        self.location = [0,0]  # [left=-1, right=1]
        self.omega = omega_0    # 0 to 1; 0.5 is equal arbitration
        self.reward = [0,0]
        self.alpha = [0,0]
        self.gamma = [0,0]
        self.action = [0,0]
        self.feedback_phase = 0
        self.cue_phase = 0
        self.load_phase = 1
    def set_cue(self, bid, trial):
        sid = self.sid
        left = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['left'].to_numpy()[0]
        right = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['right'].to_numpy()[0]
        self.letter  = [-1, 1] if (left=='A' and right=='B') else [1, -1]
        self.location = [-1, 1]
        self.reward = [0,0]
        self.alpha = [0,0]
        self.gamma = [0,0]
        self.action = [0,0]
        self.cue_phase = 1
        self.feedback_phase = 0
        self.load_phase = 0
    def set_action(self, sim, net):
        # self.action = [1, 0] if sim.data[net.p_action_left][-1]>sim.data[net.p_action_right][-1] else [0,1]
        self.action = [1, 0] if sim.data[net.p_value_left][-1]>sim.data[net.p_value_right][-1] else [0,1]
    def set_reward(self, bid, trial):
        sid = self.sid
        block = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['block'].to_numpy()[0]
        correct = self.empirical.query("sid==@sid & bid==@bid & trial==@trial")['correct'].to_numpy()[0]
        if self.action==[1,0]:
            if correct=='left':
                if self.rng.uniform(0,1) < self.p_reward:    # yes rewarded for picking the better option
                    self.reward = [1,0]
                    self.alpha = [self.alpha_plus,0]
                    self.gamma = [0,self.gamma_u]
                else:                                               # not rewarded for picking the better option
                    self.reward = [-1,0]
                    self.alpha = [self.alpha_minus,0]
                    self.gamma = [0,self.gamma_u]
            else:
                if self.rng.uniform(0,1) < 1-self.p_reward:  # yes rewarded for picking the worse option 
                    self.reward = [1,0]
                    self.alpha = [self.alpha_plus,0]
                    self.gamma = [0,self.gamma_u]
                else:                                               # not rewarded for picking the worse option
                    self.reward = [-1,0]
                    self.alpha = [self.alpha_minus,0]
                    self.gamma = [0,self.gamma_u]
        else:
            if correct=='right':
                if self.rng.uniform(0,1) < self.p_reward:    # yes rewarded for picking the better option
                    self.reward = [0,1]
                    self.alpha = [0, self.alpha_plus]
                    self.gamma = [self.gamma_u,0]
                else:                                               # not rewarded for picking the better option
                    self.reward = [0,-1]  
                    self.alpha = [0, self.alpha_minus]
                    self.gamma = [self.gamma_u,0]
            else:
                if self.rng.uniform(0,1) < 1-self.p_reward:  # yes rewarded for picking the worse option
                    self.reward = [0,1] 
                    self.alpha = [0, self.alpha_plus]
                    self.gamma = [self.gamma_u,0]
                else:                                               # not rewarded for picking the worse option   
                    self.reward = [0,-1]           
                    self.alpha = [0, self.alpha_minus]
                    self.gamma = [self.gamma_u,0]
        self.cue_phase = 0
        self.feedback_phase = 1
        self.load_phase = 0
    def set_omega(self, sim, net):
        chosen = 0 if self.action==[1,0] else 1
        v_c_letter = sim.data[net.p_value_letter][-1,chosen]
        v_c_location = sim.data[net.p_value_location][-1,chosen]
        delta_reliability = v_c_letter - v_c_location
        if delta_reliability > 0:
            self.omega += delta_reliability * self.alpha_omega * (1 - self.omega)  + self.gamma_omega * (self.omega_0 - self.omega)
        else:
            self.omega += np.abs(delta_reliability) * self.alpha_omega * (0 - self.omega) + self.gamma_omega * (self.omega_0 - self.omega)
    def sample_letter(self, t):
        return self.letter
    def sample_location(self, t):
        return self.location
    def sample_action(self, t):
        return self.action
    def sample_omega_0(self, t):
        return self.omega_0
    def sample_target_omega(self, t):
        return self.omega
    def sample_reward(self, t):
        return self.reward
    def sample_alpha(self, t):
        return self.alpha
    def sample_gamma(self, t):
        return self.gamma
    def sample_phase(self, t):
        return [self.cue_phase, self.feedback_phase, self.load_phase]

def build_network(env, n_neurons=2000, seed_network=1, inh=0, k=1.0, a=4e-5):
    net = nengo.Network(seed=seed_network)
    net.env = env
    net.config[nengo.Connection].synapse = 0.01
    net.config[nengo.Probe].synapse = 0.1
    inh_feedback = -1000*np.ones((n_neurons, 1))
    pes = nengo.PES(learning_rate=a)

    with net:
        in_letter = nengo.Node(lambda t: env.sample_letter(t))
        in_location = nengo.Node(lambda t: env.sample_location(t))
        in_omega = nengo.Node(lambda t: env.sample_omega_0(t))
        in_action = nengo.Node(lambda t: env.sample_action(t))
        in_reward = nengo.Node(lambda t: env.sample_reward(t))
        in_alpha = nengo.Node(lambda t: env.sample_alpha(t))
        in_gamma = nengo.Node(lambda t: env.sample_gamma(t))
        in_phase = nengo.Node(lambda t: env.sample_phase(t))
        target_omega = nengo.Node(lambda t: env.sample_target_omega(t))
        
        letter = nengo.Ensemble(n_neurons, 2, radius=2)
        location = nengo.Ensemble(n_neurons, 2, radius=2)
        omega = nengo.Ensemble(n_neurons, 1)
        # value_letter = nengo.Ensemble(n_neurons, 3, radius=3)
        # value_location = nengo.Ensemble(n_neurons, 3, radius=3)
        value_letter = nengo.Ensemble(n_neurons, 2)
        value_location = nengo.Ensemble(n_neurons, 2)
        value_left = nengo.Ensemble(n_neurons, 1)
        value_right = nengo.Ensemble(n_neurons, 1)
        reward = nengo.Ensemble(n_neurons, 2)
        error = nengo.Ensemble(n_neurons, 4)
        error_decode = nengo.Ensemble(1, 4, neuron_type=nengo.Direct())
        decay = nengo.Ensemble(n_neurons, 4, radius=3)
        load = nengo.Ensemble(n_neurons, 1)
        relax = nengo.Ensemble(n_neurons, 1)
        diff = nengo.Ensemble(n_neurons, 3, radius=2)
        delta_rel = nengo.Ensemble(n_neurons, 6)
        v_chosen = nengo.Ensemble(n_neurons, 2)
        
        nengo.Connection(in_letter, letter, synapse=None)
        nengo.Connection(in_location, location, synapse=None)
        nengo.Connection(in_reward, reward, synapse=None)

        # nengo.Connection(in_omega, load)
        # nengo.Connection(in_omega, relax)
        # nengo.Connection(load, omega)
        # nengo.Connection(relax, omega, transform=env.gamma_omega*0.1)
        # nengo.Connection(omega, load, transform=-1)
        # nengo.Connection(omega, relax, transform=-1)
        # nengo.Connection(in_phase[0], load.neurons, transform=inh_feedback)
        # nengo.Connection(in_phase[1], load.neurons, transform=inh_feedback)
        # nengo.Connection(in_phase[0], relax.neurons, transform=inh_feedback)
        # nengo.Connection(omega, omega, synapse=0.1)

        # nengo.Connection(in_action[0], delta_rel[0])
        # nengo.Connection(in_action[1], delta_rel[1])
        # nengo.Connection(value_letter[0], delta_rel[2])
        # nengo.Connection(value_letter[1], delta_rel[3])
        # nengo.Connection(value_location[0], delta_rel[4])
        # nengo.Connection(value_location[1], delta_rel[5])

        # nengo.Connection(delta_rel, v_chosen[0], function=lambda x: x[0]*x[2]+x[1]*x[3])
        # nengo.Connection(delta_rel, v_chosen[1], function=lambda x: x[0]*x[4]+x[1]*x[5])
        # nengo.Connection(v_chosen, diff[0], function=lambda x: np.abs(x[0]-x[1]))
        # nengo.Connection(v_chosen, diff[1], function=lambda x: 1 if x[0]>x[1] else 0)
        # nengo.Connection(omega, diff[2])
        # nengo.Connection(diff, omega, function=lambda x: 0.1*x[0]*(x[1]-x[2]))
        # nengo.Connection(in_phase[0], diff.neurons, transform=inh_feedback)        

        c0 = nengo.Connection(letter[0], value_letter[0], transform=0, learning_rule_type=pes)
        c1 = nengo.Connection(letter[1], value_letter[1], transform=0, learning_rule_type=pes)
        c2 = nengo.Connection(location[0], value_location[0], transform=0, learning_rule_type=pes)
        c3 = nengo.Connection(location[1], value_location[1], transform=0, learning_rule_type=pes)
        # nengo.Connection(omega, value_letter[2])
        # nengo.Connection(omega, value_location[2])

        # nengo.Connection(value_letter, value_left, function=lambda x: x[0]*x[2])
        # nengo.Connection(value_location, value_left, function=lambda x: x[0]*(1-x[2]))
        # nengo.Connection(value_letter, value_right, function=lambda x: x[1]*x[2])
        # nengo.Connection(value_location, value_right, function=lambda x: x[1]*(1-x[2]))

        nengo.Connection(value_letter[0], value_left)
        nengo.Connection(value_location[0], value_left)
        nengo.Connection(value_letter[1], value_right)
        nengo.Connection(value_location[1], value_right)

        nengo.Connection(in_phase[0], error.neurons, transform=inh_feedback, synapse=None)
        nengo.Connection(in_phase[2], error.neurons, transform=inh_feedback, synapse=None)
        nengo.Connection(in_phase[0], decay.neurons, transform=inh_feedback, synapse=None)
        nengo.Connection(in_phase[2], decay.neurons, transform=inh_feedback, synapse=None)

        nengo.Connection(value_letter[0], error[0], transform=-1)
        nengo.Connection(value_letter[1], error[1], transform=-1)
        nengo.Connection(value_location[0], error[0], transform=-1)
        nengo.Connection(value_location[1], error[1], transform=-1)
        nengo.Connection(reward[0], error[0], transform=+1)
        nengo.Connection(reward[1], error[1], transform=+1)
        nengo.Connection(in_alpha[0], error[2])
        nengo.Connection(in_alpha[1], error[3])

        nengo.Connection(error, c0.learning_rule, function=lambda x: -x[0]*x[2])
        nengo.Connection(error, c2.learning_rule, function=lambda x: -x[0]*x[2])
        nengo.Connection(error, c1.learning_rule, function=lambda x: -x[1]*x[3])
        nengo.Connection(error, c3.learning_rule, function=lambda x: -x[1]*x[3])

        nengo.Connection(error, error_decode[0], function=lambda x: -x[0]*x[2])
        nengo.Connection(error, error_decode[2], function=lambda x: -x[0]*x[2])
        nengo.Connection(error, error_decode[1], function=lambda x: -x[1]*x[3])
        nengo.Connection(error, error_decode[3], function=lambda x: -x[1]*x[3])

        # nengo.Connection(value_letter[0], decay[0], transform=-1)
        # nengo.Connection(value_letter[1], decay[1], transform=-1)
        # nengo.Connection(value_location[0], decay[0], transform=-1)
        # nengo.Connection(value_location[1], decay[1], transform=-1)
        # nengo.Connection(in_gamma[0], decay[2])
        # nengo.Connection(in_gamma[1], decay[3])

        # nengo.Connection(decay, c0.learning_rule, function=lambda x: -x[0]*x[2])
        # nengo.Connection(decay, c2.learning_rule, function=lambda x: -x[0]*x[2])
        # nengo.Connection(decay, c1.learning_rule, function=lambda x: -x[1]*x[3])
        # nengo.Connection(decay, c3.learning_rule, function=lambda x: -x[1]*x[3])
    
        net.p_letter = nengo.Probe(letter)
        net.p_location = nengo.Probe(location)
        net.p_value_letter = nengo.Probe(value_letter)
        net.p_value_location = nengo.Probe(value_location)
        net.p_value_left = nengo.Probe(value_left)
        net.p_value_right = nengo.Probe(value_right)
        net.p_reward = nengo.Probe(reward)
        net.p_error = nengo.Probe(error)
        net.p_error_decode = nengo.Probe(error_decode)
        net.p_decay = nengo.Probe(decay)
        net.p_v_chosen = nengo.Probe(v_chosen)
        net.p_delta_rel = nengo.Probe(delta_rel)
        net.p_diff = nengo.Probe(diff)
        net.p_omega = nengo.Probe(omega)
        net.p_omega_target = nengo.Probe(target_omega)

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
            print(f"running sid {env.sid}, block {bid}")
            for trial in env.empirical.query("sid==@sid & bid==@bid")['trial'].unique():
                # print(f"running sid {env.sid}, block {bid}, trial {trial}")
                env.set_cue(bid, trial)
                sim.run(env.t_cue)
                env.set_action(sim, net)
                env.set_omega(sim, net)
                env.set_reward(bid, trial)
                sim.run(env.t_reward)
                block = env.empirical.query("sid==@sid & bid==@bid & trial==@trial")['block'].to_numpy()[0]
                correct = env.empirical.query("sid==@sid & bid==@bid & trial==@trial")['correct'].to_numpy()[0]
                if (env.action==[1,0] and correct=='left') or (env.action==[0,1] and correct=='right'):
                    accuracy = 1
                else:
                    accuracy = 0
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
	data.to_pickle(f"data/model0_sid{sid}_behavior.pkl")
	print(data)