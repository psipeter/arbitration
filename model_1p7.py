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

def simulate(seed, monkey, session, block, trials, config='fixed'):
    params = get_params(seed, monkey, session, block, trials, config)
    net = build_network(params)
    sim = nengo.Simulator(net, progress_bar=False)
    data_list = []
    with sim:
        for trial in range(1, params['trials']+1):
            if trial==1 or trial%10==0: print(f"trial {trial}")
            set_nodes('iti', net, params, trial, sim.data)
            sim.run(params['t_iti'])
            set_nodes('cue', net, params, trial, sim.data)
            sim.run(params['t_cue'])
            set_nodes('rew', net, params, trial, sim.data)
            data_list.append(get_data(sim, net, params, trial))
            sim.run(params['t_rew'])
    dataframe = pd.DataFrame(data_list)
    # dataframe_full = get_data_full(sim, net, params)
    dataframe_full = None
    return dataframe, dataframe_full, sim, net
    # return sim, net

def get_params(seed, monkey, session, block, trials=80, config='fixed'):
    params = {
        'monkey':monkey,
        'session':session,
        'block':block,
        'trials':trials,
        'seed':seed,
        'seed_net':seed,
        'seed_rew':seed,
        # 'seed_net':int(hashlib.md5(f"{N}_{monkey}_{session}".encode()).hexdigest(), 16) % (2**32),
        # 'seed_rew':int(hashlib.md5(f"{N}_{monkey}_{session}_{block}".encode()).hexdigest(), 16) % (2**32),
        't_iti':1.0,
        't_cue':1.0,
        't_rew':1.0,
        'p_rew':0.7,
        'lr_let':3e-6,
        'lr_loc':2e-6,
        'lr_w':1e-5,
        'ramp':1.0,
        'thr':0.5,
        'w0':0.5,
        'neurons':2000,
        'tau_ff':0.02,
        'tau_p':0.02,
        'tau_fb':0.1,
    }
    if config=='fixed':
        params_net = {
            'alpha_v':0.5,
            'gamma_v':1.0,
            'alpha_w':0.4,
        }
    elif config=='random':
        rng_net = np.random.default_rng(seed=params['seed_net'])
        params_net = {
            'alpha_v':rng_net.uniform(0.4, 0.6),
            'gamma_v':rng_net.uniform(0.8, 1.0),
            'alpha_w':rng_net.uniform(0.5, 0.7),
            # 'w0':rng_net.uniform(0.49, 0.51),
        }
    params = params | params_net  # combine two parameter dictionaries
    return params

def set_nodes(phase, net, params, trial, data):
    if phase=='iti':
        net.cue.set(False, None)
        net.mask_learn.set(False, None, None)
        net.mask_decay.set(False, None, None)
        net.dec.set(False)
        net.rew.set(False, None, None)
    elif phase=='cue':
        net.cue.set(True, trial)
        net.dec.set(True)
    elif phase=='rew':
        action = data[net.p_dec][-1,0]
        net.mask_learn.set(True, trial, action)
        net.mask_decay.set(True, trial, action)
        net.rew.set(True, trial, action)

def get_data(sim, net, params, trial):
    data = {
        'seed':params['seed'],
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
        'dec':sim.data[net.p_dec][-1,0],
        'tdec':sim.data[net.p_dec][-1,1],
        # 'rew':sim.data[net.p_rew][-1,0],  # probe doesn't update in time
        # 'acc':sim.data[net.p_rew][-1,3],  # probe doesn't update in time
        'rew':net.rew.state[0],
        'acc':net.rew.state[3],
        }
    return data

def get_data_full(sim, net, params):
    n_steps =  sim.trange().shape[0]
    data = {
        'monkey':   [params['monkey']] * n_steps,
        'session':  [params['session']] * n_steps,
        'block':    [params['block']] * n_steps,
        'time':     sim.trange(),
        'va':       sim.data[net.p_v][:, 0],
        'vb':       sim.data[net.p_v][:, 1],
        'vl':       sim.data[net.p_v][:, 2],
        'vr':       sim.data[net.p_v][:, 3],
        'al':       sim.data[net.p_a][:, 0],
        'ar':       sim.data[net.p_a][:, 1],
        'w':        sim.data[net.p_w][:, 0],
        'dec':      sim.data[net.p_dec][:, 0],
        'tdec':      sim.data[net.p_dec][:, 1],
        'rew':      sim.data[net.p_rew][:, 0],
        'acc':      sim.data[net.p_rew][:, 3],
    }
    return pd.DataFrame(data)

def build_network(params):
                
    class CueNode(nengo.Node):
        def __init__(self, params, size_in=0, size_out=1):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='cue')
        def set(self, go, trial):
            if go:
                self.state[0] = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
            else:
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
            self.t_rew = params['t_rew']
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='cue')
        def step(self, t):
            t_since_cue = t % (self.t_iti + self.t_cue + self.t_rew) - self.t_iti
            if t_since_cue < 0:
                self.state[0] = self.thr
            elif t_since_cue > self.t_cue:
                self.state[0] = self.thr
            else:
                self.state[0] = self.thr * (1 - 1.2*t_since_cue/self.t_cue)
            return self.state

    class DecisionNode(nengo.Node):
        def __init__(self, params, size_in=2, size_out=2):
            self.go = False
            self.t_cue = params['t_cue']
            self.t_iti = params['t_iti']
            self.t_rew = params['t_rew']
            self.state = np.zeros((size_out))
            self.t_dec = None
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='action')
        def set(self, go):
            self.go = go  # decision period has started
            self.state = np.zeros((self.size_out))
        def step(self, t, x):
            cL, cR = x[0], x[1]
            nonzero = np.abs(cL-cR) > 0.01
            t_dec = t % (self.t_iti + self.t_cue + self.t_rew) - self.t_iti
            if self.go and nonzero and self.state[0]==0 and t_dec>0.1:  # make a new choice once
                if cL>cR: self.state[0] = 1
                elif cR>cL: self.state[0] = -1
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
            self.go = False
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='reward')
        def set(self, go, trial, action):
            self.go = go
            if go:
                cor_loc = 1 if self.emp.query("trial==@trial")['cor_loc'].values[0]=='left' else -1
                deliver = self.rng.uniform(0,1) <= self.p_rew
                if action==cor_loc and deliver:
                    self.state[0] = 1  # yes rewarded for picking the better option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = 1  # chose correctly
                if action==cor_loc and not deliver:
                    self.state[0] = -1  # not rewarded for picking the better option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = 1  # chose correctly
                if action!=cor_loc and not deliver:
                    self.state[0] = 1  # yes rewarded for picking the worse option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = -1  # chose incorrectly
                if action!=cor_loc and deliver:
                    self.state[0] = -1  # not rewarded for picking the worse option
                    self.state[1] = 0  # remove inhibition on error populations
                    self.state[2] = 1  # begin inhibition that resets the action integrator
                    self.state[3] = -1  # chose incorrectly
            else:
                self.state[0] = 0  # rewarded = 1, punished = -1
                self.state[1] = 1  # iti or cue phase = 1, reward pahse = 0
                self.state[2] = 0  # it or cue phase = 0, reward phase = 1
                self.state[3] = 0  # make correct decision = 1, incorrect = -1
        def step(self, t, x):
            return self.state

    class MaskLearningNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=4):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.letter_left, self.letter_right = None, None
            self.left_letter, self.right_letter = None, None
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_learn')
        def set(self, go, trial, action):
            if go:
                left_letter = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
                right_letter = 1 if self.emp.query("trial==@trial")['right'].values[0]=='A' else -1
                self.state = np.zeros((self.size_out))
                self.state[0] = 1 if (action==1 and left_letter==1) or (action==-1 and right_letter==1) else 0
                self.state[1] = 1 if (action==1 and left_letter==-1) or (action==-1 and right_letter==-1) else 0
                self.state[2] = 1 if action==1 else 0
                self.state[3] = 1 if action==-1 else 0
            else:
                self.state = np.zeros((self.size_out))
        def step(self, t, x):
            return self.state

    class MaskDecayNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=4):
            monkey, session, block = params['monkey'], params['session'], params['block']
            self.emp = pd.read_pickle("data/empirical2.pkl").query("monkey==@monkey & session==@session & block==@block")
            self.left_letter, self.right_letter = None, None
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def set(self, go, trial, action):
            if go:
                left_letter = 1 if self.emp.query("trial==@trial")['left'].values[0]=='A' else -1
                right_letter = 1 if self.emp.query("trial==@trial")['right'].values[0]=='A' else -1
                self.state = np.zeros((self.size_out))
                self.state[0] = 1 if (action==1 and left_letter==1) or (action==-1 and right_letter==1) else 1
                self.state[1] = 1 if (action==1 and left_letter==-1) or (action==-1 and right_letter==-1) else 1
                self.state[2] = 1 if action==1 else 1
                self.state[3] = 1 if action==-1 else 1
            else:
                self.state = np.zeros((self.size_out))
        def step(self, t, x):
            return self.state

    class MaskErrorNode(nengo.Node):
        def __init__(self, size_in=8, size_out=4):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def step(self, t, x):
            errors = x[:4]  # error in [vA, vB, vL, vR]
            masks = x[4:]  # mask for error in [vA, vB, vL, vR]
            self.state = masks * errors
            return self.state

    class VChoNode(nengo.Node):
        def __init__(self, size_in=8, size_out=2):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_decay')
        def step(self, t, x):
            vA, vB, vL, vR = x[0], x[1], x[2], x[3]
            mA, mB, mL, mR = x[4], x[5], x[6], x[7]  # indicates the chosen variables
            self.state[0] = mA*vA + mB*vB  # Q chosen for let
            self.state[1] = mL*vL + mR*vR  # Q chosen for loc
            return self.state    

    net = nengo.Network(seed=params['seed_net'])
    with net:
        # INPUTS
        cue = CueNode(params)  # input trial-specific [+/- 1] if A on L/R
        vlet = VLetNode()  # takes vA, vB as input, outputs vLetL, vLetR
        vcho = VChoNode()  # takes values as input, outputs [vChoLet, vChoLoc]
        rew = RewardNode(params)  # if action has been chosen, return trial-specific reward +/-1, and a generic reward signal: [signed_rew, abs_rew]
        mask_learn = MaskLearningNode(params)  # mask signal used to update the chosen values and locations: [mA, mB, mL, mR]
        mask_decay = MaskDecayNode(params)  # mask signal used to update the unchosen values and locations: [mA, mB, mL, mR] = 1 - mask_learn
        dec = DecisionNode(params)  # decides whether action values cross action threshold
        athr = ThrNode(params)  # inputs the dynamic action threshold, which linearly decreases from thr to 0 during t_cue
        mask_chosen = MaskErrorNode()
        mask_unchosen = MaskErrorNode()
        
        # ENSEMBLES
        f = nengo.Ensemble(params['neurons'], 2)  # letter value features
        g = nengo.Ensemble(params['neurons'], 1)  # omega features
        v = nengo.Ensemble(params['neurons'], 4)  # learned values: [vA, vB, vL, vR]
        w = nengo.Ensemble(params['neurons'], 1)  # learned omega [w]
        a = nengo.Ensemble(params['neurons'], 2)  # accumulated action values [aL, aR]
        afb = nengo.Ensemble(params['neurons'], 2)  # gate for feedback: inhibited during reward [aL, aR]
        ch = nengo.Ensemble(params['neurons'], 2, encoders=nengo.dists.Choice([[1,0],[0,1]]), intercepts=nengo.dists.Uniform(0.01, 1.0))
        vwa = nengo.Ensemble(params['neurons'], 5, radius=2)  # combined value and omega population: [vLetL, vLetR, vL, vR, w]
        evc = nengo.Ensemble(params['neurons'], 4, radius=2)  # combined error vector for chosen option: [evA, evB, evL, evR]
        evu = nengo.Ensemble(params['neurons'], 4, radius=2)  # combined error vector for unchosn option: [evA, evB, evL, evR]
        drel = nengo.Ensemble(params['neurons'], 2)  # represents [vChoLet, vChoLoc], computes difference for omega update
        ew = nengo.Ensemble(params['neurons'], 3, radius=2)  # represents all variables needed up update omega, computes the error [drel, wtar, w]

        # CONNECTIONS
        # connect feature fectors to value populations and establish the learning connections
        clet = nengo.Connection(f, v[:2], synapse=params['tau_ff'], function=lambda x: [0,0], learning_rule_type=nengo.PES(learning_rate=params['lr_let']))
        cloc = nengo.Connection(f, v[2:], synapse=params['tau_ff'], function=lambda x: [0,0], learning_rule_type=nengo.PES(learning_rate=params['lr_loc']))
        cw = nengo.Connection(g, w, synapse=params['tau_ff'], function=lambda x: params['w0'], learning_rule_type=nengo.PES(learning_rate=params['lr_w']))

        # combine all values and omega into one population
        nengo.Connection(v[:2], vlet[:2], synapse=params['tau_ff'])  # [vA, vB]
        nengo.Connection(cue, vlet[2], synapse=None)  # [+/-1] if A on L/R
        nengo.Connection(vlet, vwa[:2], synapse=None)  # [vLetL, vLetR]
        nengo.Connection(v[2:4], vwa[2:4], synapse=params['tau_ff'])  # [vL, vL]
        nengo.Connection(w, vwa[4], synapse=params['tau_ff'])  # [w]

        # compute the overall action values using the arbitration weight
        nengo.Connection(vwa, a[0], synapse=params['tau_ff'], transform=params['ramp'], function=lambda x: x[0]*x[4]+x[2]*(1-x[4]))  # vLetL*w + vL*(1-w)
        nengo.Connection(vwa, a[1], synapse=params['tau_ff'], transform=params['ramp'], function=lambda x: x[1]*x[4]+x[3]*(1-x[4]))  # vLetR*w + vR*(1-w)

        # recurrently connect the action population so that it ramps at a rate proportional to the weighted values
        nengo.Connection(a, afb, synapse=params['tau_fb'])  # action integrator
        nengo.Connection(afb, a, synapse=params['tau_fb'])  # integrate before action, decay after action
        nengo.Connection(rew[2], afb.neurons, transform=-1000*np.ones((params['neurons'], 1)), synapse=None)  # inhibition controls feedback based on phase

        # send ramping action values to a choice population that is under dynamic inhibition
        nengo.Connection(athr, ch, synapse=None, transform=[[-1],[-1]])  # send dynamic threshold to action population
        nengo.Connection(a, ch, synapse=params['tau_ff'], function=lambda x: [x[0]-x[1], x[1]-x[0]])  # send action value difference to choice population
        nengo.Connection(ch, dec, synapse=params['tau_ff'])  # send choice values to decision node

        # compute error for chosen values following choice and reward
        nengo.Connection(v, mask_chosen[:4], synapse=params['tau_ff'])  # [vA, vB, vL, vR]
        nengo.Connection(rew[0], mask_chosen[:4], transform=[[-1],[-1],[-1],[-1]], synapse=None)  # [-rew, -rew, -rew, -rew]
        nengo.Connection(mask_learn, mask_chosen[4:], synapse=None)  # [mA, mB, mL, mR]
        nengo.Connection(mask_chosen, evc, synapse=None)

        # compute error for unchosen values following choice and reward
        nengo.Connection(v, mask_unchosen[:4], synapse=params['tau_ff'], transform=-1)   # [vA, vB, vL, vR]
        nengo.Connection(mask_decay, mask_unchosen[4:], synapse=None)  # [mA, mB, mL, mR]
        nengo.Connection(mask_unchosen, evu, synapse=None)

        # compute error for omega following choice and reward
        nengo.Connection(v, vcho[:4], synapse=params['tau_ff'])   # [vA, vB, vL, vR]
        nengo.Connection(mask_learn, vcho[4:8], synapse=None)  # [mA, mB, mL, mR]
        nengo.Connection(vcho, drel, synapse=None)
        nengo.Connection(drel, ew[0], synapse=params['tau_ff'], function=lambda x: np.abs(x[0]-x[1]))  # abs(drel)
        nengo.Connection(drel, ew[1], synapse=params['tau_ff'], function=lambda x: 1 if (x[0]-x[1])>0 else 0)  # wtar = 1 if drel>0 else 0
        nengo.Connection(w, ew[2], synapse=params['tau_ff'])

        # computed errors drive PES learning
        nengo.Connection(evc[:2], clet.learning_rule, synapse=params['tau_ff'], transform=params['alpha_v'])  # learning
        nengo.Connection(evc[2:], cloc.learning_rule, synapse=params['tau_ff'], transform=params['alpha_v'])  # learning
        nengo.Connection(evu[:2], clet.learning_rule, synapse=params['tau_ff'], transform=-params['gamma_v'])  # decay
        nengo.Connection(evu[2:], cloc.learning_rule, synapse=params['tau_ff'], transform=-params['gamma_v'])  # decay
        nengo.Connection(ew, cw.learning_rule, synapse=params['tau_ff'], transform=-params['alpha_w'], function=lambda x: x[0]*(x[1]-x[2]))  # omega learning: dw = drel*(wtar-w)

        # inhibit learning and reset unless a reward is being delivered
        nengo.Connection(rew[1], evc.neurons, transform=-1000*np.ones((params['neurons'], 1)), synapse=None)
        nengo.Connection(rew[1], evu.neurons, transform=-1000*np.ones((params['neurons'], 1)), synapse=None)
        nengo.Connection(rew[1], ew.neurons, transform=-1000*np.ones((params['neurons'], 1)), synapse=None)
        
        # probes
        net.p_v = nengo.Probe(v, synapse=params['tau_p'])
        net.p_w = nengo.Probe(w, synapse=params['tau_p'])
        net.p_a = nengo.Probe(a, synapse=params['tau_p'])
        net.p_afb = nengo.Probe(afb, synapse=params['tau_p'])
        net.p_ch = nengo.Probe(ch, synapse=params['tau_p'])
        net.p_dec = nengo.Probe(dec, synapse=None)
        net.p_vlet = nengo.Probe(vlet, synapse=None)
        net.p_vwa = nengo.Probe(vwa, synapse=params['tau_p'])
        net.p_evc = nengo.Probe(evc, synapse=params['tau_p'])
        net.p_evu = nengo.Probe(evu, synapse=params['tau_p'])
        net.p_ew = nengo.Probe(ew, synapse=params['tau_p'])
        net.p_drel = nengo.Probe(drel, synapse=params['tau_p'])
        net.p_cue = nengo.Probe(cue, synapse=None)
        net.p_rew = nengo.Probe(rew, synapse=None)
        net.p_thr = nengo.Probe(athr, synapse=None)
        net.p_mask_learn = nengo.Probe(mask_learn, synapse=None)
        net.p_mask_decay = nengo.Probe(mask_decay, synapse=None)
        net.s_vwa = nengo.Probe(vwa.neurons, synapse=None)
        net.s_a = nengo.Probe(a.neurons, synapse=None)

        net.cue = cue
        net.dec = dec
        net.rew = rew
        net.mask_learn = mask_learn
        net.mask_decay = mask_decay
    
        return net

if __name__ == "__main__":
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    block = int(sys.argv[3])
    seed = int(sys.argv[4])
    config = 'random'
    s = time.time()
    nef_data, nef_data_full, sim, net = simulate(seed, monkey, session, block, trials=80, config='random')
    nef_data.to_pickle(f"data/nef/{seed}_{monkey}_{session}_{block}.pkl")
    # nef_data_full.to_pickle(f"data/nef/{monkey}_{session}_{block}_full.pkl")
    e = time.time()
    print(f"runtime (min): {(e-s)/60:.4}")
