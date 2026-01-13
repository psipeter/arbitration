import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nengo
import pandas as pd
import sys
import time
import pickle
import h5py
import hashlib

def simulate(monkey, session, block, trials, config='fixed'):
    params = get_params(monkey, session, block, trials, config)
    net = build_network(params)
    sim = nengo.Simulator(net, progress_bar=False)
    data_list = []
    with sim:
        for trial in range(1, params['trials']+1):
            if trial==1 or trial%10==0: print(f"trial {trial}")
            reset_nodes(net, params, trial)
            sim.run(params['t_iti'])
            set_nodes(net, params, trial)
            sim.run(params['t_cue'])
            data = get_data(sim, net, params, trial)
            data_list.append(data)
    dataframe = pd.DataFrame(data_list)
    return dataframe, sim, net
    # return sim, net

def get_params(monkey, session, block, trials=80, config='fixed'):
    params = {
        'monkey':monkey,
        'session':session,
        'block':block,
        'trials':trials,
        'seed_net':int(hashlib.md5(f"{monkey}_{session}".encode()).hexdigest(), 16) % (2**32),
        'seed_rew':int(hashlib.md5(f"{monkey}_{session}_{block}".encode()).hexdigest(), 16) % (2**32),
        't_iti':0.5,
        't_cue':1.5,
        'p_rew':0.7,
        # 'lr_let':3e-5,
        # 'lr_loc':0e-5,
        'lr_v':4e-5,
        'lr_w':4e-5,
        'ramp':0.1,
        'thr': 1.0,
        'neurons':1000,
    }
    if config=='fixed':
        params_net = {
            'alpha_v':0.5,
            'gamma_v':1.0,
            'alpha_w':0.4,
            'gamma_w':0.1,
        }
    elif config=='random':
        rng_net = np.random.default_rng(seed=params['seed_net'])
        params_net = {
            'alpha_v':rng_net.uniform(0.4, 0.6),
            'gamma_v':rng_net.uniform(0.9, 1.0),
            'w0':rng_net.uniform(0.49, 0.51),
            'alpha_w':rng_net.uniform(0.4, 0.6),
            'gamma_w':rng_net.uniform(0.05, 0.10),
        }
    params = params | params_net  # combine two parameter dictionaries
    return params

def set_nodes(net, params, trial):
    net.cue.set(trial)
    net.mask_learn.set(trial)
    net.mask_decay.set(trial)
    net.rew.set(trial)
    net.act.set(True)
def reset_nodes(net, params, trial):
    net.cue.reset()
    net.mask_learn.reset()
    net.mask_decay.reset()
    net.act.set(False)

def get_data(sim, net, params, trial):
    data = {
        'monkey':params['monkey'],
        'session':params['session'],
        'block':params['block'],
        'trial':trial,
        'va':sim.data[net.p_v][-1,0],
        'vb':sim.data[net.p_v][-1,1],
        'vl':sim.data[net.p_v][-1,2],
        'vr':sim.data[net.p_v][-1,3],
        'w':sim.data[net.p_w][-1,0],
        'al':sim.data[net.p_a][-1,0],
        'ar':sim.data[net.p_a][-1,1],
        'w':sim.data[net.p_w][-1,0],
        'act':sim.data[net.p_act][-1,0],
        'rew':sim.data[net.p_rew][-1,0],
        'acc':sim.data[net.p_rew][-1,3],
        'tdec':sim.data[net.p_tdec][-1,0],
    }
    return data

def build_network(params):
                
    class CueNode(nengo.Node):
        def __init__(self, params, size_in=0, size_out=1):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='cue')
        def set(self, trial):
            self.state[0] = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
        def reset(self):
            self.state = np.zeros((self.size_out))
        def step(self, t):
            return self.state

    class VLetNode(nengo.Node):
        def __init__(self, size_in=4, size_out=2):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='action')
        def step(self, t, x):
            vA, vB = x[0], x[1]  # learned values
            cue = x[2]  # 1 if A is on L, -1 if A is on R
            if cue==1:
                self.state[0] = vA
                self.state[1] = vB
            elif cue==-1:
                self.state[0] = vB
                self.state[1] = vA
            else:
                self.state = np.zeros((self.size_out))
            return self.state

    class ThrNode(nengo.Node):
        def __init__(self, params, size_in=0, size_out=1):
            self.thr = params['thr']
            self.t_cue = params['t_cue']
            self.t_iti = params['t_iti']
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='cue')
        def step(self, t):
            t_since_cue = t % (self.t_iti + self.t_cue) - self.t_iti
            if t_since_cue < 0:
                self.state[0] = self.thr
            else:
                self.state[0] = self.thr * (1 - 1.1*t_since_cue/self.t_cue)
            return self.state

    class ActionNode(nengo.Node):
        def __init__(self, params, size_in=3, size_out=2):
            self.go = False
            self.thr = params['thr']
            self.t_cue = params['t_cue']
            self.t_iti = params['t_iti']
            self.state = np.zeros((size_out))
            self.t_dec = None
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='action')
        def set(self, go):
            self.go = go  # decision period has started
            self.state = np.zeros((self.size_out))
        def step(self, t, x):
            aL, aR = x[0], x[1]
            thr = x[2]
            if self.go and np.abs(aL-aR) > thr and self.state[0]==0 :  # make a new choice once
                t_dec = t % (self.t_iti + self.t_cue) - self.t_iti
                if aL>aR: self.state[0] = 1
                elif aR>aL: self.state[0] = -1
                else: self.state[0] = 0
                self.state[1] = t_dec  # decision (reaction) time
            return self.state

    class RewardNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=4):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.p_rew = params['p_rew']
            self.rng = np.random.default_rng(seed=params['seed_rew'])
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.cor_loc = None
            self.deliver = None
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='reward')
        def set(self, trial):
            self.cor_loc = self.emp.query("trial==@trial")['cor_loc'].values[0]
            self.deliver = self.rng.uniform(0,1) <= self.p_rew
        def step(self, t, x):
            action = x[0]
            if action==0:
                self.state[0] = 0  # rewarded = 1, punished = -1
                self.state[1] = 1  # cue/decision phase = 1, reward pahse = 0
                self.state[2] = 0  # cue/decision phase = 0, reward pahse = 1
                self.state[3] = 0  # make correct decision = 1, incorrect = -1
            else:
                cor_loc = 1 if self.cor_loc=='left' else -1
                if action==cor_loc and self.deliver:
                    self.state[0] = 1  # yes rewarded for picking the better option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = 1  # chose correctly
                if action==cor_loc and not self.deliver:
                    self.state[0] = -1  # not rewarded for picking the better option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = 1  # chose correctly
                if action!=cor_loc and not self.deliver:
                    self.state[0] = 1  # yes rewarded for picking the worse option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = -1  # chose incorrectly
                if action!=cor_loc and self.deliver:
                    self.state[0] = -1  # not rewarded for picking the worse option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = -1  # chose incorrectly
            return self.state

    class MaskLearningNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=4):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.letter_left, self.letter_right = None, None
            self.left_letter, self.right_letter = None, None
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_learn')
        def set(self, trial):
            self.left_letter = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
            self.right_letter = 1 if self.emp.query("trial==@trial")['right'].values[0]=='A' else -1
        def reset(self):
            self.left_letter, self.right_letter = None, None
        def step(self, t, x):
            action = x[0]  # 1 if left is chosen, -1 if right is chosen
            self.state = np.zeros((self.size_out))
            self.state[0] = 1 if (action==1 and self.left_letter==1) or (action==-1 and self.right_letter==1) else 0
            self.state[1] = 1 if (action==1 and self.left_letter==-1) or (action==-1 and self.right_letter==-1) else 0
            self.state[2] = 1 if action==1 else 0
            self.state[3] = 1 if action==-1 else 0
            return self.state
            # [vA, vB, vL, vR]: 1 if learning should occur because letter/loc was chosen, 0 otherwise

    class MaskDecayNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=4):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.left_letter, self.right_letter = None, None
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def set(self, trial):
            self.left_letter = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
            self.right_letter = 1 if self.emp.query("trial==@trial")['right'].values[0]=='A' else -1
        def reset(self):
            self.left_letter, self.right_letter = None, None
        def step(self, t, x):
            action = x[0]  # 1 if left is chosen, -1 if right is chosen
            self.state = np.ones((self.size_out))
            self.state[0] = 0 if (action==1 and self.left_letter==1) or (action==-1 and self.right_letter==1) else 1
            self.state[1] = 0 if (action==1 and self.left_letter==-1) or (action==-1 and self.right_letter==-1) else 1
            self.state[2] = 0 if action==1 else 1
            self.state[3] = 0 if action==-1 else 1
            return self.state
            # [vA, vB, vL, vR]: 1 if decay should occur because letter/loc was NOT chosen, 0 otherwise

    class ApplyMaskNode(nengo.Node):
        def __init__(self, size_in=9, size_out=4):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def step(self, t, x):
            errors = x[:4]  # error in [vA, vB, vL, vR]
            masks = x[4:8]  # mask for error in [vA, vB, vL, vR]
            stop = x[8]  # turn learning off during iti/cue phase
            if stop==1:
                self.state = np.zeros((self.size_out))
            else:
                self.state = masks * errors
            return self.state

    class WNode(nengo.Node):
        def __init__(self, params, size_in=10, size_out=1):
            self.w0 = params['w0']
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def step(self, t, x):
            vA, vB, vL, vR = x[0], x[1], x[2], x[3]
            mA, mB, mL, mR = x[4], x[5], x[6], x[7]  # indicates the chosen variables
            stop = x[8]  # turn learning off during iti/cue phase
            w = x[9]  # current decoded omega
            if stop==1:
                self.state = np.zeros((self.size_out))
            else:
                drel = (mA*vA+mB*vB) - (mL*vL+mR*vR)  # dQ_chosen_stim - dQ_chosen_choice
                wtar = int(drel>0)  # 1/0
                self.state[0] = np.abs(drel) * (wtar - w) 
            return self.state

    net = nengo.Network(seed=params['seed_net'])
    with net:
        # INPUTS
        in_f = nengo.Node([0,0,0,0])  # constant vector that provides features for learning values
        in_g = nengo.Node([0])  # constant vector that provides features for learning omega
        # in_w = nengo.Node(params['w0'])  # baseline omega
        cue = CueNode(params)  # input trial-specific [+/- 1] if A on L/R
        vlet = VLetNode()  # takes vA, vB as input, outputs vLetL, vLetR
        rew = RewardNode(params)  # if action has been chosen, return trial-specific reward +/-1, and a generic reward signal: [signed_rew, abs_rew]
        mask_learn = MaskLearningNode(params)  # mask signal used to update the chosen values and locations: [mA, mB, mL, mR]
        mask_decay = MaskDecayNode(params)  # mask signal used to update the unchosen values and locations: [mA, mB, mL, mR] = 1 - mask_learn
        act = ActionNode(params)  # decides whether action values cross action threshold
        athr = ThrNode(params)  # inputs the dynamic action threshold, which linearly decreases from thr to 0 during t_cue
        evc = ApplyMaskNode()
        evu = ApplyMaskNode()
        ewt = WNode(params)
        
        # ENSEMBLES
        f = nengo.Ensemble(params['neurons'], 4)  # value features
        g = nengo.Ensemble(params['neurons'], 1)  # omega features
        v = nengo.Ensemble(params['neurons'], 4)  # learned values: [vA, vB, vL, vR]
        w = nengo.Ensemble(params['neurons'], 1)  # learned omega [w]
        a = nengo.Ensemble(params['neurons'], 2)  # accumulated action values [aL, aR]
        # afb = nengo.Ensemble(params['neurons'], 2)  # gate for feedback: inhibited during reward [aL, aR]
        # vlet = nengo.Ensemble(params['neurons'], 4)  # learned values for letters, masked by letter location on current trial [vA, vB, mL, mR]
        vwa = nengo.Ensemble(params['neurons'], 5, radius=2)  # combined value and omega population: [vLetL, vLetR, vL, vR, w]
        # evc = nengo.Ensemble(params['neurons'], 8, radius=4)  # combined error vector for chosen option and mask: [vA-E, vB-E, vL-E, vR-E, mA, mB, mL, mR]
        # evu = nengo.Ensemble(params['neurons'], 8, radius=4)  # combined error vector for unchosn option and mask: [vA-E, vB-E, vL-E, vR-E, mA, mB, mL, mR]
        drel = nengo.Ensemble(params['neurons'], 8, radius=4)  # combined value vector for chosen option and mask, for updaing omega: [vA, vB, vL, vR, mA, mB, mL, mR]
        wtar = nengo.Ensemble(params['neurons'], 1)  # target omega value following action and reward  [wtar]
        # ewt = nengo.Ensemble(params['neurons'], 1)  # error for omega update  [ew]
        # ewd = nengo.Ensemble(params['neurons'], 1)  # error for omega decay  [ew]

        # CONNECTIONS
        # connect feature fectors to value populations and establish the learning connections
        nengo.Connection(in_f, f)
        nengo.Connection(in_g, g)
        # cf = nengo.Connection(f[:2], v[:2], synapse=0.01, transform=0, learning_rule_type=nengo.PES(learning_rate=params['lr_let']))  # learned connection for letters
        # cf2 = nengo.Connection(f[2:], v[2:], synapse=0.01, transform=0, learning_rule_type=nengo.PES(learning_rate=params['lr_loc']))  # learned connection for locations
        cf = nengo.Connection(f, v, synapse=0.01, transform=0, learning_rule_type=nengo.PES(learning_rate=params['lr_v']))
        cg = nengo.Connection(g, w, synapse=0.01, function=lambda x: params['w0'], learning_rule_type=nengo.PES(learning_rate=params['lr_w']))  # learned connection for omega

        # combine all values and omega into one population
        nengo.Connection(v[:2], vlet[:2], synapse=0.01)  # [vA, vB]
        nengo.Connection(cue, vlet[2], synapse=None)  # [+/-1] if A on L/R
        nengo.Connection(vlet, vwa[:2], synapse=None)  # [vLetL, vLetR]
        # nengo.Connection(cue, vlet[2:4])  # [1,0] if [A,B] or [0,1] if [B,A], serves as a mask for value routing in the network
        # nengo.Connection(vlet, vwa[0], synapse=0.02, function=lambda x: x[0]*x[2]+x[1]*x[3])  # computes vLetL using above mask
        # nengo.Connection(vlet, vwa[1], synapse=0.02, function=lambda x: x[1]*x[2]+x[0]*x[3])  # computes vLetR using above mask
        nengo.Connection(v[2:4], vwa[2:4], synapse=0.01)  # [vL, vL]
        nengo.Connection(w, vwa[4], synapse=0.01)  # [w]

        # compute the overall action values using the arbitration weight
        nengo.Connection(vwa, a[0], synapse=0.01, transform=params['ramp'], function=lambda x: x[0]*x[4]+x[2]*(1-x[4]))  # vLetL*w + vL*(1-w)
        nengo.Connection(vwa, a[1], synapse=0.01, transform=params['ramp'], function=lambda x: x[1]*x[4]+x[3]*(1-x[4]))  # vLetR*w + vR*(1-w)

        # recurrently connect the action population so that it ramps at a rate proportional to the weighted values
        # nengo.Connection(a, afb, synapse=0.1)  # action integrator
        # nengo.Connection(afb, a, synapse=0.1)  # integrate before action, decay after action
        # nengo.Connection(rew[2], afb.neurons, transform=-1000*np.ones((params['neurons'], 1)), synapse=None)  # inhibition controls feedback based on phase
        nengo.Connection(a, act[:2], synapse=0.01)  # send action values to action node
        nengo.Connection(athr, act[2], synapse=None)  # send dynamic threshold to action node
        nengo.Connection(act[0], rew, synapse=None)  # send [+/-1] to reward node
        nengo.Connection(act[0], mask_learn, synapse=None)  # send [+/-1] to learning mask node
        nengo.Connection(act[0], mask_decay, synapse=None)  # send [+/-1] to decay mask node

        # compute error for value following choice and reward
        nengo.Connection(v, evc[:4], synapse=0.01)  # [vA, vB, vL, vR]
        nengo.Connection(rew[0], evc[:4], transform=4*[[-1]])  # [-rew, -rew, -rew, -rew]
        nengo.Connection(mask_learn, evc[4:8])  # [mA, mB, mL, mR]
        nengo.Connection(v, evu[:4], synapse=0.01, transform=-1)   # [vA, vB, vL, vR]
        nengo.Connection(mask_decay, evu[4:8])  # [mA, mB, mL, mR]

        # compute error for omega following choice and reward
        # nengo.Connection(v, drel[:4], synapse=0.01)   # [vA, vB, vL, vR]
        # nengo.Connection(mask_learn, drel[4:8])  # [mA, mB, mL, mR]
        # nengo.Connection(drel, wtar, synapse=0.01, function=lambda x: [x[0]*x[4]+x[1]*x[5]-x[2]*x[6]-x[3]*x[7]])  # target omega based on delta reliability
        # nengo.Connection(wtar, ewt, synapse=0.01, function=lambda x: 1 if x>0 else 0)  # error for omega = 1-w if drel>0 else 0-w
        # nengo.Connection(w, ewt, synapse=0.01, transform=-1)  # omega current
        # nengo.Connection(in_w, ewd)  # decay back towards w0
        # nengo.Connection(w, ewd, synapse=0.01, transform=-1)  # decay proportional to omega current
        nengo.Connection(v, ewt[:4], synapse=0.01)   # [vA, vB, vL, vR]
        nengo.Connection(mask_learn, ewt[4:8])  # [mA, mB, mL, mR]
        nengo.Connection(w, ewt[9], synapse=0.01)  # omega current



        # computed errors drive PES learning
        # nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=params['alpha_v'], function=lambda x: [x[0]*x[4], x[1]*x[5]])  # let learning
        # nengo.Connection(evc, cf2.learning_rule, synapse=0.01, transform=params['alpha_v'], function=lambda x: [x[2]*x[6], x[3]*x[7]])  # loc learning
        # nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-params['gamma_v'], function=lambda x: [x[0]*x[4], x[1]*x[5]])  # let decay
        # nengo.Connection(evu, cf2.learning_rule, synapse=0.01, transform=-params['gamma_v'], function=lambda x: [x[2]*x[6], x[3]*x[7]])  # loc decay
        # nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=params['alpha_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])  # let learning
        # nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-params['gamma_v'], function=lambda x: [x[0]*x[4], x[1]*x[5], x[2]*x[6], x[3]*x[7]])  # let decay
        # nengo.Connection(ewt, cg.learning_rule, synapse=0.01, transform=-params['alpha_w'])  # omega learning
        # nengo.Connection(ewd, cg.learning_rule, synapse=0.01, transform=-params['gamma_w'])  # omega decay
        nengo.Connection(evc, cf.learning_rule, synapse=0.01, transform=params['alpha_v'])  # let learning
        nengo.Connection(evu, cf.learning_rule, synapse=0.01, transform=-params['gamma_v'])  # let decay
        nengo.Connection(ewt, cg.learning_rule, synapse=0.01, transform=-params['alpha_w'])  # omega learning

        # inhibit learning and reset unless a reward is being delivered
        # nengo.Connection(rew[1], evc.neurons, transform=-1000*np.ones((params['neurons'], 1)))
        # nengo.Connection(rew[1], evu.neurons, transform=-1000*np.ones((params['neurons'], 1)))
        # nengo.Connection(rew[1], ewt.neurons, transform=-1000*np.ones((params['neurons'], 1)))
        # nengo.Connection(rew[1], ewd.neurons, transform=-1000*np.ones((params['neurons'], 1)))
        nengo.Connection(rew[1], evc[8], synapse=None)
        nengo.Connection(rew[1], evu[8], synapse=None)
        nengo.Connection(rew[1], ewt[8], synapse=None)
        
        # probes
        net.p_v = nengo.Probe(v, synapse=0.01)
        net.p_w = nengo.Probe(w, synapse=0.01)
        net.p_a = nengo.Probe(a, synapse=0.01)
        # net.p_afb = nengo.Probe(afb, synapse=0.01)
        net.p_act = nengo.Probe(act[0])
        net.p_tdec = nengo.Probe(act[1], synapse=None)
        net.p_vlet = nengo.Probe(vlet, synapse=None)
        vwaout = nengo.Ensemble(1, 2, neuron_type=nengo.Direct())  # readout of vLetL and vLetR
        nengo.Connection(vwa, vwaout[0], synapse=0.01, transform=params['ramp'], function=lambda x: x[0]*x[4]+x[2]*(1-x[4]))  # vLetL*w + vL*(1-w)
        nengo.Connection(vwa, vwaout[1], synapse=0.01, transform=params['ramp'], function=lambda x: x[1]*x[4]+x[3]*(1-x[4]))  # vLetR*w + vR*(1-w)
        net.p_vwa = nengo.Probe(vwa, synapse=0.01)
        net.p_vwaout = nengo.Probe(vwaout)
        net.p_evc = nengo.Probe(evc)
        net.p_evu = nengo.Probe(evu)
        net.p_ewt = nengo.Probe(ewt)
        # net.p_ewd = nengo.Probe(ewd)
        net.p_drel = nengo.Probe(drel)
        net.p_cue = nengo.Probe(cue)
        net.p_rew = nengo.Probe(rew)
        net.p_mask_learn = nengo.Probe(mask_learn)
        net.p_mask_decay = nengo.Probe(mask_decay)
        net.s_vwa = nengo.Probe(vwa.neurons, synapse=None)
        net.s_a = nengo.Probe(a.neurons, synapse=None)

        net.cue = cue
        net.act = act
        net.rew = rew
        net.mask_learn = mask_learn
        net.mask_decay = mask_decay
    
        return net

if __name__ == "__main__":
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    block = int(sys.argv[3])
    config = 'random'
    s = time.time()
    nef_data, sim, net = simulate(monkey, session, block, trials=80, config='random')
    nef_data.to_pickle(f"data/nef/{monkey}_{session}_{block}.pkl")
    e = time.time()
    print(f"runtime (min): {(e-s)/60:.4}")