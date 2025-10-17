import numpy as np
import scipy
import pandas as pd
import sys
import time
import optuna
from rl_model import RL_env, RL_model, output_data #, run_rl_model
import matplotlib.pyplot as plt

def curve_rmse_loss(monkey, rl_data):
	emp = pd.read_pickle("data/empirical.pkl").query("monkey==@monkey")
	emp['trial_pre_reversal'] = np.where(emp['trial'] < emp['reversal_at_trial'], emp['trial'], None)
	emp['trial_post_reversal'] = np.where(emp['trial'] >= emp['reversal_at_trial'], emp['trial'] - emp['reversal_at_trial'], None)
	emp['model_type'] = 'monkey'
	emp.rename(columns={'monkey_choice': 'cloc'}, inplace=True)
	emp.rename(columns={'monkey_accuracy': 'acc'}, inplace=True)
	emp.rename(columns={'reward': 'rew'}, inplace=True)
	# emp['rew'] = emp['rew'].replace(0, -1)
	emp['seed'] = 'empirical'
	emp = emp.drop(columns=['left', 'right', 'correct', 'reversal_at_trial'])

	rl_data["model_type"] = "rl"
	combined = pd.concat([rl_data, emp], ignore_index=True)

	model_types = ['monkey', 'rl']
	block_types = ['what', 'where']
	phases = {'pre': 'trial_post_reversal', 'post': 'trial_pre_reversal'}
	accs = []
	for model in model_types:
		for block in block_types:
			for phase_name, phase_col in phases.items():
				# Filter the data for this condition
				sub_df = combined[
					(combined['model_type'] == model) &
					(combined['block_type'] == block) &
					(combined[phase_col] > 0) &
					(combined[phase_col] <= 30)
				].copy()
				grouped = sub_df.groupby([phase_col, 'block', 'seed'])['acc'].mean()
				trial_means = grouped.groupby(level=0).mean()
				for trial, acc in trial_means.items():
					accs.append({'model_type': model,'block_type': block,'phase': phase_name,'trial': trial,'mean_acc': acc})
	acc_df = pd.DataFrame(accs)

	rmses = []
	blocks = ['what', 'where']
	phases = ['pre', 'post']
	for block in block_types:
		for phase in ['pre', 'post']:
			monkey_curve = acc_df.query("model_type=='monkey' & block_type==@block & phase==@phase")['mean_acc'].to_numpy()
			rl_curve = acc_df.query("model_type=='rl' & block_type==@block & phase==@phase")['mean_acc'].to_numpy()
			# print(block, phase, monkey_curve.shape, rl_curve.shape)
			rmse = np.sqrt(np.mean((monkey_curve - rl_curve)**2))
			rmses.append(pd.DataFrame([[block, phase, rmse]], columns=['block', 'phase', 'rmse']))
	losses = pd.concat(rmses, ignore_index=True)
	total_loss = losses['rmse'].sum()
	return total_loss

# def NLL_loss(rl_data, monkey, params):
# 	NLLs = []
# 	emp = pd.read_pickle("data/empirical.pkl").query("monkey==@monkey")
# 	for session in emp['session'].unique():
# 		for block in emp.query("session==@session")['block'].unique():
# 			for trial in emp.query("session==@session & block==@block")['trial'].unique():
# 				pls = rl_data.query("session==@session & block==@block & trial==@trial")['pl'].to_numpy()
# 				prs = rl_data.query("session==@session & block==@block & trial==@trial")['pr'].to_numpy()
# 				print(session, block, trial, prs)
# 				monkey_choice = emp.query("session==@session & block==@block & trial==@trial")['cloc'].values[0]
# 				if monkey_choice=='left':
# 					NLL = -np.log(pls.mean())
# 				elif monkey_choice=='right':
# 					NLL = -np.log(prs.mean())
# 				if NLL:
# 					NLLs.append(NLL)
# 	return np.mean(NLLs)

def NLL_loss(rl_data, monkey, session):
	emp = pd.read_pickle("data/empirical.pkl").query("monkey == @monkey & session==@session")
	rl_agg = rl_data.groupby(["session", "block", "trial"], as_index=False)[["pl", "pr"]].mean()
	merged = emp.merge(rl_agg, on=["session", "block", "trial"], how="inner")
	eps = 1e-10  # Small value to avoid log(0)
	merged["pl_safe"] = merged["pl"].clip(eps, 1 - eps)
	merged["pr_safe"] = merged["pr"].clip(eps, 1 - eps)
	merged["NLL"] = np.where(merged["cloc"] == "left", -np.log(merged["pl_safe"]), -np.log(merged["pr_safe"]))
	return merged["NLL"].mean()

def train_rl(optuna_trial, monkey, session, test=False, fitted_params=None):
	session_block = []
	emp = pd.read_pickle("data/empirical.pkl").query("monkey==@monkey")
	blocks = emp.query("session==@session")['block'].unique()
	if test:
		params = fitted_params.iloc[0].to_dict()
	else:
		params = {
			'beta':optuna_trial.suggest_float('beta', 1e1, 1e2, step=1.0),
			'alpha_plus':optuna_trial.suggest_float('alpha_plus', 0.3, 0.7, step=0.01),
			'alpha_minus':optuna_trial.suggest_float('alpha_minus', 0.3, 0.7, step=0.01),
			'gamma_u':optuna_trial.suggest_float('gamma_u', 0.1, 0.5, step=0.01),
			'w0':optuna_trial.suggest_float('w0', 0.3, 0.7, step=0.01),
			'alpha_w':optuna_trial.suggest_float('alpha_w', 0.2, 0.6, step=0.01),
			'gamma_w':optuna_trial.suggest_float('gamma_w', 0.02, 0.2, step=0.01),
		}
	dfs = []
	for b, block in enumerate(blocks):
		print(f"session {session}, block {block}")
		seed = block+100*session+1000 if monkey=='W' else block+100*session
		trials = emp.query("session==@session & block==@block")['trial'].unique()
		env = RL_env(monkey, session, block, seed=seed)
		model = RL_model(params, env, seed=seed)
		for trial in trials:
			env.set_cue(trial)
			model.act()
			env.set_reward(trial, model)
			model.update(env.reward)
			df = output_data(trial, env, model)
			dfs.append(df)
	rl_data = pd.concat(dfs, ignore_index=True)
	if test:
		rl_data.to_pickle(f"data/rl/monkey{monkey}_session{session}_values.pkl")
	# loss = NLL_loss(rl_data, monkey, session)
	loss = curve_rmse_loss(monkey, rl_data)
	return loss

def fit_rl(monkey, session, optuna_trials=1000):
	study = optuna.create_study(direction="minimize")
	study.optimize(lambda optuna_trial: train_rl(optuna_trial, monkey, session), n_trials=optuna_trials)
	best_params = study.best_params
	best_loss = study.best_value
	best_params['loss'] = best_loss
	best_df = pd.DataFrame([best_params])
	best_df.to_pickle(f"data/{monkey}_{session}_params.pkl")
	print(best_df)
	best_params = pd.read_pickle(f"data/{monkey}_{session}_params.pkl")
	test_loss = train_rl(None, monkey, session, test=True, fitted_params=best_params)
	print(f"test loss: {test_loss}")


if __name__ == '__main__':
	monkey = sys.argv[1]
	session = int(sys.argv[2])
	print(f"fitting monkey {monkey} session {session}")
	start = time.time()
	fit_rl(monkey, session)
	end = time.time()
	print(f"runtime {(end-start)/60:.4} min")