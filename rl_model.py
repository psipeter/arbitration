import numpy as np
import scipy
import pandas as pd
import sys

class RL_env():
	def __init__(self, monkey, session, block, p_reward=0.7, fitting=False, reward_seed=None):
		self.monkey = monkey
		self.block = block
		self.p_reward = p_reward
		self.fitting = fitting
		self.empirical = pd.read_pickle("data/empirical.pkl")
		self.reward_seed = reward_seed if reward_seed is not None else (block+100*session+100000 if monkey=='W' else block+100*session)
		self.rng = np.random.RandomState(seed=reward_seed)
		self.correct = None
		self.reward = None
		self.accuracy = None
		if self.fitting:
			self.block_type = 'what' if block<= 12 else 'where'
			self.correct_let = 'A' if self.rng.uniform(0,1)<0.5 else 'B'
			self.correct_loc = 'left' if self.rng.uniform(0,1)<0.5 else 'right'
			self.reversal_at_trial = self.rng.randint(30, 51)
			self.session = reward_seed
		else:
			self.session = session
			try:
				self.reversal_at_trial = self.empirical.query("monkey==@monkey & session==@session & block==@block")['reversal_at_trial'].unique()[0]
			except:
				pass
	def set_cue(self, trial):
		monkey = self.monkey
		session = self.session
		block = self.block
		if self.fitting:
			if trial==self.reversal_at_trial:
				self.correct_let = 'B' if self.correct_let=='A' else 'A'
				self.correct_loc = 'right' if self.correct_loc=='left' else 'left'
			self.left= 'A' if self.rng.uniform(0,1)<0.5 else 'B'
			self.right = 'B' if self.left=='A' else 'A'
		else:
			self.left = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['left'].to_numpy()[0]
			self.right = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['right'].to_numpy()[0]
	def set_reward(self, trial, action):
		monkey = self.monkey
		session = self.session
		block = self.block
		if self.fitting:
			correct = 'left' if ((self.block_type=='where' and self.correct_loc=='left') or (self.block_type=='what' and self.correct_let==self.left)) else 'right'
		else:
			correct = self.empirical.query("monkey==@monkey & session==@session & block==@block & trial==@trial")['correct'].to_numpy()[0]
		deliver_reward = self.rng.uniform(0,1)
		if (action=='left' and correct=='left') or (action=='right' and correct=='right'):
			accuracy = 1
			if deliver_reward<=self.p_reward:
				reward = 1  # yes rewarded for picking the better option
			else:
				reward = 0  # not rewarded for picking the better option
		elif (action=='left' and correct=='right') or (action=='right' and correct=='left'):
			accuracy = 0
			if deliver_reward<=(1-self.p_reward):
				reward = 1  # yes rewarded for picking the worse option
			else:
				reward = 0  # not rewarded for picking the worse option
		else:
			raise
		self.correct = correct
		self.reward = reward
		self.accuracy = accuracy


class RL_model():
	def __init__(self, params, env):
		self.params = params
		self.env = env
		self.va = 0
		self.vb = 0
		self.vl = 0  # V_Act1
		self.vr = 0  # V_Act2
		self.w = params['w0']
		self.w0 = params['w0']
		self.al = None
		self.ar = None
		self.cloc = None
		self.clet = None
	def act(self):
		self.vletl = self.va if self.env.left=='A' else self.vb  # V_StimL
		self.vletr = self.va if self.env.right=='A' else self.vb  # V_StimR
		self.al = self.vletl*self.w + self.vl*(1-self.w)  # DV_left
		self.ar = self.vletr*self.w + self.vr*(1-self.w)  # DV_right
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
		alpha = self.params['alpha_plus'] if reward==1 else self.params['alpha_minus']
		# gamma = self.params['gamma_u']
		# gamma = alpha
		unreward = 0 if reward==1 else 1
		for attr in chosen:
			val = getattr(self, attr)
			updated = val + alpha * (reward - val)
			setattr(self, attr, updated)
		for attr in unchosen:
			val = getattr(self, attr)
			updated = val + alpha * (unreward - val)  # equal and opposite update for unchosen
			# updated = val + gamma * (0 - val)  # decay value for unchosen
			setattr(self, attr, updated)
		# update omega
		drel = getattr(self, chosen[0]) - getattr(self, chosen[1])
		wtar = 1 if drel>0 else 0
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
		'reward_seed',
		'block_type',
		'trial_pre_reversal',
		'trial_post_reversal',
		'model_type',
		'letl',
		'va',
		'vb',
		'vl',
		'vr',
		'w',
		'al',
		'ar',
		'clet',
		'cloc',
		'rew',
		'acc']
	df = pd.DataFrame([[
		env.monkey,
		env.session,
		env.block,
		trial,
		env.reward_seed,
		block_type,
		trial_pre_reversal,
		trial_post_reversal,
		'rl',
		env.left,
		model.va,
		model.vb,
		model.vl,
		model.vr,
		model.w,
		model.al,
		model.ar,
		model.clet,
		model.cloc,
		env.reward,
		env.accuracy
		]],columns=columns)
	return df

def run_rl_model(monkey, session, block, params, session_config):
	if session_config=='empirical':
		env = RL_env(monkey, session, block)
		trials = env.empirical.query("monkey==@monkey & session==@session & block==@block")['trial'].unique()
	else:
		env = RL_env(monkey, session, block, reward_seed=session, fitting=True)
		trials = np.arange(1,81)
	model = RL_model(params, env)
	for trial in trials:
		env.set_cue(trial)
		model.act()
		env.set_reward(trial, model.cloc)
		model.update(env.reward)
		df = output_data(trial, env, model)
		df.to_pickle(f"data/rl/monkey{monkey}_session{session}_block{block}_trial{trial}_values.pkl")

if __name__ == "__main__":
	# monkey = sys.argv[1]
	# session = int(sys.argv[2])
	# param_config = sys.argv[3]
	param_config = sys.argv[1]
	session_config = sys.argv[2]
	sessions = [0,1,2,3] if session_config=='empirical' else range(300)
	for monkey in ['V', 'W']:
		for session in sessions:
			if param_config=='load':
				params = pd.read_pickle(f"data/rl/{monkey}_params.pkl").iloc[0].to_dict()
			elif param_config=='random':
				rng = np.random.RandomState(seed = session + 4 if monkey=='W' else session)
				params = {
					'alpha_plus':rng.uniform(0.4, 0.6),
					'alpha_minus':rng.uniform(0.4, 0.6),
					# 'gamma_u':rng.uniform(0.4, 0.6),
					'w0':rng.uniform(0.3, 0.6),
					'alpha_w':rng.uniform(0.5, 0.8),
					'gamma_w':rng.uniform(0.01, 0.05),
				}
			else:
				print("Must specify which parameters to use")
				raise

			blocks = 24
			for block in range(1, blocks+1):
				run_rl_model(monkey, session, block, params, session_config)