import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
import time
import optuna
from model_1 import run_to_fit

def choice_loss(monkey, session, nef_data):
	emp_data = pd.read_pickle("data/empirical.pkl").query("monkey==@monkey & session==@session")
	nef_choice = nef_data['choice'].to_numpy()
	emp_choice = emp_data['monkey_choice'].to_numpy()
	print('emp', emp_choice)
	print('nef', nef_choice)
	if emp_choice.shape != nef_choice.shape:
		raise ValueError("Choice arrays must have the same shape.")
	loss = np.sum(emp_choice != nef_choice) / emp_choice.size
	print(loss)
	return loss

def model1_loss(trial, monkey, session):
    alpha_chosen = trial.suggest_float("alpha_chosen", 0.01, 1.0, step=0.01)
    alpha_unchosen = trial.suggest_float("alpha_unchosen", 0.01, 1.0, step=0.01)
    omega_0 = trial.suggest_float("omega_0", 0.01, 1.0, step=0.01)
    alpha_omega = trial.suggest_float("alpha_omega", 0.01, 1.0, step=0.01)
    gamma_omega = trial.suggest_float("gamma_omega", 0.01, 1.0, step=0.01)
    neurons = trial.suggest_int("neurons", 3000, 3000, step=1)
    data = run_to_fit(monkey, session, alpha_chosen, alpha_unchosen, omega_0, alpha_omega, gamma_omega, neurons)
    loss = choice_loss(monkey, session, data)
    return loss

def fit_model1(monkey, session, optuna_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: model1_loss(trial, monkey, session), n_trials=optuna_trials)
    best_params = study.best_trial.params
    loss = study.best_trial.value
    param_names = ["monkey", "session"]
    params = [monkey, session]
    for key, value in best_params.items():
        param_names.append(key)
        params.append(value)
    print(f"{len(study.trials)} trials completed. Best value is {loss:.4}")
    performance_data = pd.DataFrame([[monkey, session, loss]], columns=['monkey', 'session', 'loss'])
    performance_data.to_pickle(f"data/{monkey}_{session}_performance.pkl")
    fitted_params = pd.DataFrame([params], columns=param_names)
    fitted_params.to_pickle(f"data/{monkey}_{session}_params.pkl")
    return performance_data, fitted_params

if __name__ == '__main__':
    monkey = sys.argv[1]
    session = int(sys.argv[2])
    print(f"fitting monkey {monkey}, session {session}")
    start = time.time()
    performance_data, fitted_params = fit_model1(monkey, session)
    end = time.time()
    print(performance_data)
    print(fitted_params)
    print(f"runtime {(end-start)/60:.4} min")