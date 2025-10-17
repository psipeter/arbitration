import numpy as np
import scipy
import pandas as pd
import sys

class RL_env():
	def __init__(self, monkey, session, block, seed, p_reward=0.7):
		self.monkey = monkey
		self.session = session
		self.block = block
		self.p_reward = p_reward
		self.empirical = pd.read_pickle("data/empirical.pkl")
		self.seed = seed
		self.rng = np.random.RandomState(seed=seed)
		self.correct = None
		self.reward = None
		self.accuracy = None
	def set_cue(self, trial):
		monkey = self.monkey
		session = self.session
		block = self.block
		self.reversal_at_trial = self.empirical.query("monkey==@monkey & session==@session & block==@block")['reversal_at_trial'].unique()[0]
		self.left = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['left'].to_numpy()[0]
		self.right = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['right'].to_numpy()[0]
	def set_reward(self, trial, model):
		monkey = self.monkey
		session = self.session
		block = self.block
		action = model.cloc
		correct = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['correct'].to_numpy()[0]
		deliver_reward = self.rng.uniform(0,1)
		if (action=='left' and correct=='left') or (action=='right' and correct=='right'):
			# accuracy = 1
			if deliver_reward<=self.p_reward:
				reward = 1  # yes rewarded for picking the better option
			else:
				reward = -1  # not rewarded for picking the better option
		elif (action=='left' and correct=='right') or (action=='right' and correct=='left'):
			# accuracy = 0
			if deliver_reward<=(1-self.p_reward):
				reward = 1  # yes rewarded for picking the worse option
			else:
				reward = -1  # not rewarded for picking the worse option
		else:
			raise
		self.correct = correct
		self.reward = reward
		self.accuracy = model.pl if correct=='left' else model.pr


class RL_model():
	def __init__(self, params, env, seed):
		self.params = params
		self.env = env
		self.rng = np.random.RandomState(seed=seed)
		self.va = 0.0
		self.vb = 0.0
		self.vl = 0.0  # V_Act1
		self.vr = 0.0  # V_Act2
		self.w0 = params['w0']
		self.w = self.w0
		self.al = None
		self.ar = None
		self.pl = None
		self.pr = None
		self.cloc = None
		self.clet = None
	def act(self):
		self.vletl = self.va if self.env.left=='A' else self.vb  # V_StimL
		self.vletr = self.va if self.env.right=='A' else self.vb  # V_StimR
		self.al = self.vletl*self.w + self.vl*(1-self.w)  # DV_left
		self.ar = self.vletr*self.w + self.vr*(1-self.w)  # DV_right
		beta = self.params['beta']
		if beta:
			self.pl, self.pr = scipy.special.softmax(beta*np.array([self.al, self.ar]))[:]
			# self.pl, self.pr = 0.5, 0.5
			# self.pl = self.rng.uniform(0,1)
			# self.pr = 1-self.pl
			if self.rng.uniform(0,1)<self.pl:
				self.cloc = 'left'
				self.clet = self.env.left
			else:
				self.cloc = 'right'
				self.clet = self.env.right
		else:  # deterministic choice
			self.pl = 1 if self.al>self.ar else 0
			self.pr = 1 if self.ar>self.al else 0
			self.cloc = 'left' if self.al>self.ar else 'right'
			self.clet = 'A' if (self.cloc=='left' and self.env.left=='A') or (self.cloc=='right' and self.env.right=='A') else 'B'
	def update(self, reward):
		# update values
		chosen, unchosen = [], []
		if self.clet == 'A':
			chosen.append('va')
			unchosen.append('vb')
		elif self.clet == 'B':
			chosen.append('vb')
			unchosen.append('va')
		if self.cloc == 'left':
			chosen.append('vl')
			unchosen.append('vr')
		elif self.cloc == 'right':
			chosen.append('vr')
			unchosen.append('vl')
		# gamma = self.params['gamma']
		# beta = self.params['beta']
		alpha = self.params['alpha_plus'] if reward==1 else self.params['alpha_minus']
		# gamma = self.params['gamma_u']
		# gamma = alpha
		# unreward = 0 if reward==1 else 1
		for attr in chosen:
			val = getattr(self, attr)
			updated = val + alpha * (reward - val)
			setattr(self, attr, updated)
		for attr in unchosen:
			val = getattr(self, attr)
			# updated = val + alpha * (unreward - val)  # equal and opposite update for unchosen
			# updated = val + gamma * (0 - val)  # decay value for unchosen
			updated = (1 - self.params['gamma_u']) * val
			setattr(self, attr, updated)
		# update omega
		drel = getattr(self, chosen[0]) - getattr(self, chosen[1])
		wtar = 1 if drel>0 else 0
		# self.w = self.w + beta * np.abs(drel) * (wtar-self.w)
		self.w = self.w + self.params['alpha_w']*np.abs(drel)*(wtar-self.w) + self.params['gamma_w']*(self.w0 - self.w)

def output_data(trial, env, model):
	block_type = 'what' if env.block<12 else 'where'
	trial_pre_reversal = trial if trial<env.reversal_at_trial else -1
	trial_post_reversal = trial - env.reversal_at_trial if trial>=env.reversal_at_trial else -1
	columns = [
		'monkey',
		'session',
		'block',
		'trial',
		'block_type',
		'trial_pre_reversal',
		'trial_post_reversal',
		'model_type',
		'seed',
		'letl',
		'va',
		'vb',
		'vl',
		'vr',
		'w',
		'al',
		'ar',
		'pl',
		'pr',
		'clet',
		'cloc',
		'rew',
		'acc']
	df = pd.DataFrame([[
		env.monkey,
		env.session,
		env.block,
		trial,
		block_type,
		trial_pre_reversal,
		trial_post_reversal,
		'rl',
		env.seed,
		env.left,
		model.va,
		model.vb,
		model.vl,
		model.vr,
		model.w,
		model.al,
		model.ar,
		model.pl,
		model.pr,
		model.clet,
		model.cloc,
		env.reward,
		env.accuracy
		]],columns=columns)
	return df

def run_rl_model(monkey, session, block, params):
	seed = block+100*session+1000 if monkey=='W' else block+100*session
	env = RL_env(monkey, session, block, seed=seed)
	model = RL_model(params, env, seed=seed)
	trials = env.empirical.query("monkey==@monkey & session==@session & block==@block")['trial'].unique()
	for trial in trials:
		env.set_cue(trial)
		model.act()
		env.set_reward(trial, model)
		model.update(env.reward)
		df = output_data(trial, env, model)
		df.to_pickle(f"data/rl/monkey{monkey}_session{session}_block{block}_trial{trial}_values.pkl")

if __name__ == "__main__":
	param_config = sys.argv[1]
	sessions = [0,1,2,3]
	for monkey in ['V', 'W']:
		for session in sessions:
			if param_config=='load':
				params = pd.read_pickle(f"data/rl_params/{monkey}_params.pkl").iloc[0].to_dict()
				# params['beta'] = None
				print(params)
			else:
				# params = {
				# 	'alpha_plus':0.47,
				# 	'alpha_minus':0.44,
				# 	'gamma_u':0.32,
				# 	'w0':0.5,
				# 	'alpha_w':0.31,
				# 	'gamma_w':0.09,
				# 	'beta': 20,
				# }
				# params = {'beta': 2.27, 'alpha_plus': 0.6, 'alpha_minus': 0.36, 'gamma_u': 0.240, 'w0': 0.35, 'alpha_w': 0.22, 'gamma_w': 0.15}
				# params = {'beta': 329.552193143731, 'alpha_plus': 0.69, 'alpha_minus': 0.32999999999999996, 'gamma_u': 0.21000000000000002, 'w0': 0.31, 'alpha_w': 0.43000000000000005, 'gamma_w': 0.18000000000000002}
				params = {'beta': 1.6973305195890587, 'alpha_plus': 0.35, 'alpha_minus': 0.48, 'gamma_u': 0.32, 'w0': 0.69, 'alpha_w': 0.24000000000000002, 'gamma_w': 0.12}
			for block in range(1, 25):
				run_rl_model(monkey, session, block, params)