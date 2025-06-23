import numpy as np
from scipy.io import savemat

for monkey in ['V', 'W']:
	for session in [0,1,2,3]:
		spike_dict = {'value': [], 'omega': [], 'action': [], 'mixed': [], 'error': [], 'reliability': []}
		value_dict = {'a': [], 'b': [], 'l': [], 'r': [], 'wab': [], 'wlr': [], 'al': [], 'ar': [], 'acc': [], 'rew': [], 'clet': [], 'cloc': []}
		for bid in range(1, 25):
			try:
				data = np.load(f"data/spikes/monkey{monkey}_session{session}_block{bid}_spikes.npz")
				spike_dict['value'].append(data['v'])  # value population encodes value of A, B, L, and R together
				spike_dict['omega'].append(data['w'])  # omega population encodes omega and 1-omega
				spike_dict['action'].append(data['a'])  # action population encodes the final value of choose L and choose R
				spike_dict['mixed'].append(data['m'])  # mixed population encodes A, B, L, R, omega, 1-omega
				spike_dict['error'].append(data['e'])  # error population encodes the prediction error for the value of the chosen option
				spike_dict['reliability'].append(data['r'])  # drel population encodes delta_reliability
			except:
				print('spikes missing', monkey, session, bid)
			try:
				data = np.load(f"data/spikes/monkey{monkey}_session{session}_block{bid}_values.npz")
				value_dict['a'].append(data['va'])  # value of A
				value_dict['b'].append(data['vb'])  # value of B
				value_dict['l'].append(data['vl'])  # value of L
				value_dict['r'].append(data['vr'])  # value of R
				value_dict['wab'].append(data['wab'])  # omega
				value_dict['wlr'].append(data['wlr'])  # 1-omega
				value_dict['al'].append(data['al'])  # final value of choose L
				value_dict['ar'].append(data['ar'])  # final value of chosoe R
				value_dict['acc'].append(data['acc'])  # did the model choose correctly ([0,1])
				value_dict['rew'].append(data['rew'])  # was the model rewarded ([0,1])
				value_dict['clet'].append(data['clet'])  # what letter did the model choose ([0,1])
				value_dict['cloc'].append(data['cloc'])  # what location did the model choose ([0,1])
			except:
				print('values missing', monkey, session, bid)
		mat_spike_dict = {}
		mat_value_dict = {}
		for key, value in spike_dict.items():
			mat_spike_dict[key] = np.concatenate(value, axis=2).astype(np.float16)
		for key, value in value_dict.items():
			mat_value_dict[key] = np.concatenate(value).astype(np.float16)
		savemat(f"data/spikes/monkey{monkey}_session{session}_spikes.mat", mat_spike_dict, do_compression=True)  # shape = (time x neurons x trials) = (300 x 3000 x 1920)
		savemat(f"data/spikes/monkey{monkey}_session{session}_values.mat", mat_value_dict, do_compression=True)  # shape = (trials) = (1920)