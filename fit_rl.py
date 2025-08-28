import numpy as np
import scipy
import pandas as pd
import sys
import time
import optuna
from rl_model import RL_env, RL_model, output_data #, run_rl_model

def curve_rmse_loss(monkey, rl_data):
	emp = pd.read_pickle("data/empirical.pkl").query("monkey==@monkey")
	emp['trial_pre_reversal'] = np.where(emp['trial'] < emp['reversal_at_trial'], emp['trial'], None)
	emp['trial_post_reversal'] = np.where(emp['trial'] >= emp['reversal_at_trial'], emp['trial'] - emp['reversal_at_trial'], None)
	emp['model_type'] = 'monkey'
	emp.rename(columns={'monkey_choice': 'cloc'}, inplace=True)
	emp.rename(columns={'monkey_accuracy': 'acc'}, inplace=True)
	emp.rename(columns={'reward': 'rew'}, inplace=True)
	# emp['rew'] = emp['rew'].replace(0, -1)
	emp['reward_seed'] = 'empirical'
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
				grouped = sub_df.groupby([phase_col, 'block', 'reward_seed'])['acc'].mean()
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

def run_to_fit(monkey, params):
	dfs = []
	for ft in range(params['fitting_trials']):
		for block in range(1, 25):
			env = RL_env(monkey, ft, block, fitting=True, reward_seed=ft)
			model = RL_model(params, env)
			for trial in range(1, 81):
				env.set_cue(trial)
				model.act()
				env.set_reward(trial, model.cloc)
				model.update(env.reward)
				df = output_data(trial, env, model)
				dfs.append(df)
	rl_data = pd.concat(dfs, ignore_index=True)
	return rl_data


def rl_loss(trial, monkey):
	# params = {
	# 	'fitting_trials': 20,
	# 	'alpha_plus':trial.suggest_float('alpha_plus', 0.4, 0.6, step=0.01),
	# 	'alpha_minus':trial.suggest_float('alpha_minus', 0.4, 0.6, step=0.01),
	# 	'gamma_u':trial.suggest_float('gamma_u', 0.2, 0.4, step=0.01),
	# 	'w0':trial.suggest_float('w0', 0.3, 0.6, step=0.01),
	# 	'alpha_w':trial.suggest_float('alpha_w', 0.5, 0.8, step=0.01),
	# 	'gamma_w':trial.suggest_float('gamma_w', 0.01, 0.05, step=0.01),
	# }
	params = {
		'fitting_trials': 30,
		'alpha_plus':trial.suggest_float('alpha_plus', 0.1, 0.9, step=0.01),
		'alpha_minus':trial.suggest_float('alpha_minus', 0.1, 0.9, step=0.01),
		# 'gamma_u':trial.suggest_float('gamma_u', 0.1, 0.9, step=0.01),
		'w0':trial.suggest_float('w0', 0.1, 0.9, step=0.01),
		'alpha_w':trial.suggest_float('alpha_w', 0.1, 0.9, step=0.01),
		'gamma_w':trial.suggest_float('gamma_w', 0.01, 0.2, step=0.01),
	}
	data = run_to_fit(monkey, params)
	loss = curve_rmse_loss(monkey, data)
	# print(data, loss)
	return loss

def fit_rl(monkey, optuna_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: rl_loss(trial, monkey), n_trials=optuna_trials)
    best_params = study.best_trial.params
    loss = study.best_trial.value
    param_names = ["monkey"]
    params = [monkey]
    for key, value in best_params.items():
        param_names.append(key)
        params.append(value)
    print(f"{len(study.trials)} trials completed. Best value is {loss:.4}")
    performance_data = pd.DataFrame([[monkey, loss]], columns=['monkey', 'loss'])
    performance_data.to_pickle(f"data/rl/{monkey}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/rl/{monkey}_params.pkl")
    return performance_data, fitted_params

if __name__ == '__main__':
	monkey = sys.argv[1]
	print(f"fitting monkey {monkey}")
	start = time.time()
	performance_data, fitted_params = fit_rl(monkey)
	end = time.time()
	print(performance_data)
	print(fitted_params.iloc[0].to_dict())
	print(f"runtime {(end-start)/60:.4} min")