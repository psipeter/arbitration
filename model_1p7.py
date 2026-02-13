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

def simulate(monkey, session, block, seed, trials, config, pert):
    params = get_params(monkey, session, block, seed, trials, config, pert)
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
            data_list.append(get_values(sim, net, params, trial))
            sim.run(params['t_rew'])
    values = pd.DataFrame(data_list)
    probes = get_probes(sim, net, params)
    return values, probes, sim, net

def get_params(monkey, session, block, seed, trials=80, config='random', pert=0):
    params = {
        'monkey':monkey,
        'session':session,
        'block':block,
        'trials':trials,
        'seed':seed,
        # 'seed_net':seed,
        # 'seed_rew':seed,
        'seed_net':int(hashlib.md5(f"{monkey}_{session}_{block}_{seed}".encode()).hexdigest(), 16) % (2**32),
        'seed_rew':int(hashlib.md5(f"{monkey}_{session}_{block}_{seed}".encode()).hexdigest(), 16) % (2**32),
        't_iti':1.0,
        't_cue':1.0,
        't_rew':1.0,
        'p_rew':0.7,
        'r_let':0.6,
        'r_loc':0.3,
        'thr':0.7,
        'w0':0.5,
        'neurons':1000,
        'tau_ff':0.02,
        'tau_p':0.02,
        'tau_fb':0.1,
        'tau_inh':0.1,
        'pert':pert,
    }
    if config=='fixed':
        params_net = {
            'alpha_v':0.5,
            'gamma_v':1.0,
            'alpha_w':0.4,
            'lr_let':5e-6,
            'lr_loc':5e-6,
            'lr_w':5e-6,
        }
    elif config=='random':
        rng_net = np.random.default_rng(seed=params['seed_net'])
        params_net = {
            'alpha_v':rng_net.uniform(0.3, 0.6),
            'gamma_v':rng_net.uniform(0.7, 1.0),
            'alpha_w':rng_net.uniform(0.2, 0.4),
            'lr_let':rng_net.uniform(3e-6, 6e-6),
            'lr_loc':rng_net.uniform(3e-6, 6e-6),
            'lr_w':rng_net.uniform(3e-6, 7e-6),
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
        net.pert.set()
    elif phase=='cue':
        net.cue.set(True, trial)
        net.dec.set(True)
    elif phase=='rew':
        action = data[net.p_dec][-1,0]
        net.mask_learn.set(True, trial, action)
        net.mask_decay.set(True, trial, action)
        net.rew.set(True, trial, action)
        net.pert.reset()

def get_values(sim, net, params, trial):
    tidx = int(sim.data[net.p_dec][-1,3])  # timestep of decision
    data = {
        'monkey':params['monkey'],
        'session':params['session'],
        'block':params['block'],
        'seed':params['seed'],
        'trial':trial,
        'va':sim.data[net.p_v][tidx,0],
        'vb':sim.data[net.p_v][tidx,1],
        'vl':sim.data[net.p_v][tidx,2],
        'vr':sim.data[net.p_v][tidx,3],
        'w':sim.data[net.p_w][tidx,0],
        'vwa_a':sim.data[net.p_vwa][tidx,0],  # note: letter on L
        'vwa_b':sim.data[net.p_vwa][tidx,1],  # note: letter on R
        'vwa_l':sim.data[net.p_vwa][tidx,2],
        'vwa_r':sim.data[net.p_vwa][tidx,3],
        'vwa_w':sim.data[net.p_vwa][tidx,4],
        'al':sim.data[net.p_a][tidx,0],
        'ar':sim.data[net.p_a][tidx,1],
        'w':sim.data[net.p_w][tidx,0],
        'dec':sim.data[net.p_dec][tidx,0],
        'tdec':sim.data[net.p_dec][tidx,1],
        'thr':sim.data[net.p_thr][tidx,0],
        'rew':net.rew.state[0],
        'acc':net.rew.state[3],
        'pert':params['pert'],
        }
    return data

def get_probes(sim, net, params):
    n_steps =  sim.trange()[::10].shape[0]
    data = {
        'monkey':   [params['monkey']] * n_steps,
        'session':  [params['session']] * n_steps,
        'block':    [params['block']] * n_steps,
        'time':     sim.trange()[::10],
        'va':       sim.data[net.p_v][::10, 0],
        'vb':       sim.data[net.p_v][::10, 1],
        'vl':       sim.data[net.p_v][::10, 2],
        'vr':       sim.data[net.p_v][::10, 3],
        'al':       sim.data[net.p_a][::10, 0],
        'ar':       sim.data[net.p_a][::10, 1],
        'w':        sim.data[net.p_w][::10, 0],
        'dec':      sim.data[net.p_dec][::10, 0],
        'tdec':     sim.data[net.p_dec][::10, 1],
        'thr':      sim.data[net.p_thr][::10, 0],
        'rew':      sim.data[net.p_rew][::10, 0],
        'acc':      sim.data[net.p_rew][::10, 3],
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
        def __init__(self, params, size_in=2, size_out=4):
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
                self.state[2] = 1  # some action has been chosen
                self.state[3] = int(t/0.001)  # simulation timestep of decision
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
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='mask_error')
        def step(self, t, x):
            errors = x[:4]  # error in [vA, vB, vL, vR]
            masks = x[4:]  # mask for error in [vA, vB, vL, vR]
            self.state = masks * errors
            return self.state

    class VChoNode(nengo.Node):
        def __init__(self, size_in=8, size_out=2):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='vcho')
        def step(self, t, x):
            vA, vB, vL, vR = x[0], x[1], x[2], x[3]
            mA, mB, mL, mR = x[4], x[5], x[6], x[7]  # indicates the chosen variables
            self.state[0] = mA*vA + mB*vB  # Q chosen for let
            self.state[1] = mL*vL + mR*vR  # Q chosen for loc
            return self.state    

    class WTarNode(nengo.Node):
        def __init__(self, size_in=2, size_out=1):
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='wtar')
        def step(self, t, x):
            vc_let = x[0]  # value of chosen letter
            vc_loc = x[1]  # value of chosen letter
            drel = vc_let - vc_loc
            self.state[0] = 1 if drel>0 else 0
            return self.state

    class PertNode(nengo.Node):
        def __init__(self, params, size_in=1, size_out=1):
            self.t_cue = params['t_cue']
            self.t_iti = params['t_iti']
            self.t_rew = params['t_rew']
            self.pert = params['pert']
            self.state = np.zeros((size_out))
            super().__init__(self.step, size_in=size_in, size_out=size_out, label='pert')
        def reset(self):
            self.state[0] = 0
        def set(self):
            self.state[0] = self.pert
        def step(self, t, x):
            return self.state

    net = nengo.Network(seed=params['seed_net'])
    winh = -1000*np.ones((params['neurons'], 1))
    with net:
        # INPUTS
        cue = CueNode(params)  # input trial-specific [+/- 1] if A on L/R
        vlet = VLetNode()  # takes vA, vB as input, outputs vLetL, vLetR
        vcho = VChoNode()  # takes values as input, outputs [vChoLet, vChoLoc]
        wtar = WTarNode()  # takes drel as input, outputs 1 if drel>0 (letter) else 0 (location)
        rew = RewardNode(params)  # if action has been chosen, return trial-specific reward +/-1, and a generic reward signal: [signed_rew, abs_rew]
        mask_learn = MaskLearningNode(params)  # mask signal used to update the chosen values and locations: [mA, mB, mL, mR]
        mask_decay = MaskDecayNode(params)  # mask signal used to update the unchosen values and locations: [mA, mB, mL, mR] = 1 - mask_learn
        dec = DecisionNode(params)  # decides whether action values cross action threshold
        athr = ThrNode(params)  # inputs the dynamic action threshold, which linearly decreases from thr to 0 during t_cue
        mask_chosen = MaskErrorNode()
        mask_unchosen = MaskErrorNode()
        pert = PertNode(params)  # perturbs w population: [w drive]
        
        # ENSEMBLES
        f = nengo.Ensemble(params['neurons'], 2)  # letter value features
        g = nengo.Ensemble(params['neurons'], 1)  # omega features
        v = nengo.Ensemble(params['neurons'], 4, radius=0.5)  # learned values: [vA, vB, vL, vR]
        w = nengo.Ensemble(params['neurons'], 1)  # learned omega [w]
        a = nengo.Ensemble(params['neurons'], 2, radius=0.3)  # accumulated action values [aL, aR]
        ch = nengo.Ensemble(params['neurons'], 2, encoders=nengo.dists.Choice([[1,0],[0,1]]), intercepts=nengo.dists.Uniform(0.01, 1.0))
        vwa = nengo.Ensemble(params['neurons'], 5)  # combined value and omega population: [vLetL, vLetR, vL, vR, w]
        evc = nengo.Ensemble(params['neurons'], 4, radius=2)  # combined error vector for chosen option: [evA, evB, evL, evR]
        evu = nengo.Ensemble(params['neurons'], 4, radius=2)  # combined error vector for unchosn option: [evA, evB, evL, evR]
        ew = nengo.Ensemble(params['neurons'], 1)  # error in omega representation [ew]

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
        nengo.Connection(vwa, a[0], synapse=params['tau_fb'], transform=params['r_let'], function=lambda x: x[0]*x[4])  # vLetL*w
        nengo.Connection(vwa, a[1], synapse=params['tau_fb'], transform=params['r_let'], function=lambda x: x[1]*x[4])  # vLetR*w
        nengo.Connection(vwa, a[0], synapse=params['tau_fb'], transform=params['r_loc'], function=lambda x: x[2]*(1-x[4]))  # vL*(1-w)
        nengo.Connection(vwa, a[1], synapse=params['tau_fb'], transform=params['r_loc'], function=lambda x: x[3]*(1-x[4]))  # vR*(1-w)

        # recurrently connect the action population so that it ramps at a rate proportional to the weighted values
        nengo.Connection(a, a, synapse=params['tau_fb'])  # action integrator
        nengo.Connection(dec[2], a.neurons, transform=winh/1000, synapse=nengo.Alpha(params['tau_inh']))  # inhibition controls feedback if decision has been made

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
        nengo.Connection(vcho, wtar, synapse=params['tau_ff'])
        nengo.Connection(wtar, ew, synapse=None, transform=-1)
        nengo.Connection(w, ew, synapse=params['tau_ff'])

        # computed errors drive PES learning
        nengo.Connection(evc[:2], clet.learning_rule, synapse=params['tau_ff'], transform=params['alpha_v'])  # learning
        nengo.Connection(evc[2:], cloc.learning_rule, synapse=params['tau_ff'], transform=params['alpha_v'])  # learning
        nengo.Connection(evu[:2], clet.learning_rule, synapse=params['tau_ff'], transform=-params['gamma_v'])  # decay
        nengo.Connection(evu[2:], cloc.learning_rule, synapse=params['tau_ff'], transform=-params['gamma_v'])  # decay
        nengo.Connection(ew, cw.learning_rule, synapse=params['tau_ff'], transform=params['alpha_w'])  # omega learning

        # inhibit learning and reset unless a reward is being delivered
        nengo.Connection(rew[1], evc.neurons, transform=winh, synapse=None)
        nengo.Connection(rew[1], evu.neurons, transform=winh, synapse=None)
        nengo.Connection(rew[1], ew.neurons, transform=winh, synapse=None)

        # inhibit value estimate between decision time and reward presentation
        nengo.Connection(dec[2], v.neurons, transform=winh, synapse=None)  # decision inhibits value neurons
        nengo.Connection(rew[2], v.neurons, transform=-winh, synapse=None)  # decision excites value neurons, cancelling inhibition

        # perturb omega representation by driving omega towards 1 or 0
        nengo.Connection(pert, w, synapse=None)
        
        # probes
        net.p_v = nengo.Probe(v, synapse=params['tau_p'])
        net.p_w = nengo.Probe(w, synapse=params['tau_p'])
        net.p_a = nengo.Probe(a, synapse=params['tau_p'])
        # net.p_afb = nengo.Probe(afb, synapse=params['tau_p'])
        net.p_ch = nengo.Probe(ch, synapse=params['tau_p'])
        net.p_dec = nengo.Probe(dec, synapse=None)
        net.p_vlet = nengo.Probe(vlet, synapse=None)
        net.p_vwa = nengo.Probe(vwa, synapse=params['tau_p'])
        net.p_evc = nengo.Probe(evc, synapse=params['tau_p'])
        net.p_evu = nengo.Probe(evu, synapse=params['tau_p'])
        net.p_ew = nengo.Probe(ew, synapse=params['tau_p'])
        # net.p_drel = nengo.Probe(drel, synapse=params['tau_p'])
        net.p_cue = nengo.Probe(cue, synapse=None)
        net.p_rew = nengo.Probe(rew, synapse=None)
        net.p_thr = nengo.Probe(athr, synapse=None)
        net.p_wtar = nengo.Probe(wtar, synapse=None)
        net.p_mask_learn = nengo.Probe(mask_learn, synapse=None)
        net.p_mask_decay = nengo.Probe(mask_decay, synapse=None)
        net.p_pert = nengo.Probe(pert, synapse=None)
        # net.s_vwa = nengo.Probe(vwa.neurons, synapse=params['tau_p'])
        # net.s_a = nengo.Probe(a.neurons, synapse=params['tau_p'])

        net.cue = cue
        net.dec = dec
        net.rew = rew
        net.mask_learn = mask_learn
        net.mask_decay = mask_decay
        net.pert = pert
    
        return net

if __name__ == "__main__":
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    block = int(sys.argv[3])
    seed = int(sys.argv[4])
    perts = [-0.2, -0.1, 0, 0.1, 0.2]
    config = 'random'
    start = time.time()
    values = []
    probes = []
    for p in perts:
        val, pro, sim, net = simulate(monkey, session, block, seed, 80, config, p)
        values.append(val)
        probes.append(pro)
    df_values = pd.concat(values, ignore_index=True)
    df_probes = pd.concat(probes, ignore_index=True)
    df_values.to_pickle(f"data/nef/{monkey}_{session}_{block}_{seed}_values.pkl")
    df_probes.to_pickle(f"data/nef/{monkey}_{session}_{block}_{seed}_probes.pkl")
    end = time.time()
    print(f"runtime (min): {(end-start)/60:.4}")
