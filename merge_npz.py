import numpy as np
import scipy

for monkey in ['V', 'W']:
	for session in [0,1,2,3]:
		spike_dict = {'value': [], 'omega': [], 'action': [], 'mixed': [], 'error': [], 'reliability': []}
		value_dict = {'a': [], 'b': [], 'l': [], 'r': [], 'wab': [], 'wlr': [], 'al': [], 'ar': [], 'acc': []}
		for bid in range(1, 25):
			try:
				data = np.load(f"data/spikes/monkey{monkey}_session{session}_block{bid}_spikes.npz")
				spike_dict['value'].append(data['v'])
				spike_dict['omega'].append(data['w'])
				spike_dict['action'].append(data['a'])
				spike_dict['mixed'].append(data['m'])
				spike_dict['error'].append(data['e'])
				spike_dict['reliability'].append(data['r'])
				data = np.load(f"data/spikes/monkey{monkey}_session{session}_block{bid}_values.npz")
				value_dict['a'].append(data['va'])
				value_dict['b'].append(data['vb'])
				value_dict['l'].append(data['vl'])
				value_dict['r'].append(data['vr'])
				value_dict['wab'].append(data['wab'])
				value_dict['wlr'].append(data['wlr'])
				value_dict['al'].append(data['al'])
				value_dict['ar'].append(data['ar'])
				value_dict['acc'].append(data['acc'])
			except:
				print(monkey, session, bid)
		mat_spike_dict = {}
		mat_value_dict = {}
		for key, value in spike_dict.items():
			mat_spike_dict[key] = np.concatenate(value, axis=2)
		for key, value in value_dict.items():
			mat_value_dict[key] = np.concatenate(value, axis=2)
		scipy.io.savemat(f"data/spikes/monkey{monkey}_session{session}_spikes.mat", mat_spike_dict)
		scipy.io.savemat(f"data/spikes/monkey{monkey}_session{session}_values.mat", mat_value_dict)