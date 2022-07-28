from identity import *
from quantize import Quantize, QuantizeGZ
#from itcompress import *
from squish import *
from delta import *
from xor import *
from spartan import *
from apca import *
from hier import *
#from quarc_trc import *
import numpy as np

error_thresholds = [0.15,0.1,0.075,0.05,0.025,0.01,0.0075,0.005,0.0025,0.001]

#ERROR_THRESHOLD = 0.00499

def initialize(ERROR_THRESH):
	#set up baslines
	BASELINES = []
	#BASELINES.append(IdentityGZ('gz', error_thresh=ERROR_THRESH))
	#BASELINES.append(Quantize('q', error_thresh=ERROR_THRESH))
	#BASELINES.append(QuantizeGZ('q+gz', error_thresh=ERROR_THRESH))
	#BASELINES.append(ItCompress('itcmp', error_thresh=ERROR_THRESH))
	#BASELINES.append(Spartan('sptn', error_thresh=ERROR_THRESH))
	#BASELINES.append(EntropyCoding('q+ent', error_thresh=ERROR_THRESH))
	#BASELINES.append(Sprintz('spz', error_thresh=ERROR_THRESH))
	#BASELINES.append(SprintzLearnedGzip('spz+ln', error_thresh = ERROR_THRESH))
	#BASELINES.append(SprintzGzip('spz+gz', error_thresh=ERROR_THRESH))
	#BASELINES.append(Gorilla('grla', error_thresh=ERROR_THRESH))
	#BASELINES.append(GorillaLossy('grla+l', error_thresh=ERROR_THRESH))
	BASELINES.append(MultivariateHierarchical('hier', error_thresh = 0.001, blocksize=4096*4, start_level = 10, trc = True))
	#BASELINES.append(AdaptivePiecewiseConstant('apca', error_thresh=ERROR_THRESH))
	#BASELINES.append(Quarc('mts',model = 'online_VAR', error_thresh = ERROR_THRESH, trc = True, col_sel_algo = 'randomK', sample_size_factor = 1, param_error_thresh = 0.001, number_columns=18, hybrid = False, first_diff= False))
	return BASELINES

def run(BASELINES,\
        deco_t = 0.001,\
		DATA_DIRECTORY = '/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/data/', \
		FILENAME = 'bitcoin.npy',\
		):
	data = np.load(DATA_DIRECTORY + FILENAME, allow_pickle = True)[:4096*256,:]

	bresults = {}
	for nn in BASELINES:
		#print(data.shape)
		nn.load(data)
		nn.compress()
		nn.decompress(data, deco_t)
		bresults[nn.target] = nn.compression_stats

	return bresults

#plotting

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (10,4)

for t in error_thresholds:

	#ERROR_THRESHOLD = t
	print(t)
	BASELINES = initialize(t)
	FILENAME = 'new_hier.npy'
	#SIZE_LIMIT = 1000
	bresults = run(BASELINES, t)


	#compressed size
	"""
	plt.figure()
	plt.title(FILENAME.split('.')[0] + ": Compression Ratio at " + str(ERROR_THRESHOLD))
	plt.ylabel('Compression Ratio')
	plt.bar([k for k in bresults], [bresults[k]['compressed_ratio'] for k in bresults])
	plt.bar([k for k in bresults], [bresults[k].get('model_size',0)/bresults[k].get('original_size') for k in bresults])
	plt.legend(['Compression Ratio'])
	#plt.legend(['Compression Ratio', 'Model Contribution'])
	plt.savefig('compression_ratio_' +str(ERROR_THRESHOLD) + '_' + FILENAME.split('.')[0]+'.png')
	"""
	try:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_compression_r_hier_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			#f.write('[')
			[f.write(str(bresults[k]['compressed_ratio'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()

	except:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_compression_r_hier_' +FILENAME.split('.')[0] + '.txt','a') as f:
			#f.write('[')
			[f.write(str(bresults[k]['compressed_ratio'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()

	"""
	#compression throughput (subtract bitpacking time)
	plt.figure()
	plt.title(FILENAME.split('.')[0] + ": Throughput" )
	plt.ylabel('Thpt bytes/sec')

	x1 = [i - 0.1 for i,_ in enumerate(bresults)]
	x2 = [i + 0.1 for i,_ in enumerate(bresults)]
	x = [i for i,_ in enumerate(bresults)]
	labels = [k for _,k in enumerate(bresults)]

	plt.bar(x1, [bresults[k]['original_size']/(bresults[k]['compression_latency'] - bresults[k].get('bitpacktime',0))  for k in bresults], width=0.2)
	plt.bar(x2, [bresults[k]['compressed_size']/(bresults[k]['decompression_latency'])  for k in bresults], width=0.2)

	plt.xticks(x,labels)
	plt.legend(['Compression', 'Decompression'])
	plt.yscale('log')
	plt.yticks([10000,100000,1000000, 10000000,1e8])
	plt.savefig('compression_tpt_' + FILENAME.split('.')[0]+'.png')
	"""
	#decompression throughput (subtract bitpacking time)
	try:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_compression_l_hier_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			#f.write('[')
			[f.write(str(bresults[k]['compression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()
	except:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_compression_l_hier_' +FILENAME.split('.')[0] + '.txt','a') as f:
			#f.write('[')
			[f.write(str(bresults[k]['compression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()

	try:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_decompression_l_hier_' + FILENAME.split('.')[0] + '.txt', 'x') as f:
			#f.write('[')
			[f.write(str(bresults[k]['decompression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()
	except:
		with open('/Users/brunobarbarioli/Documents/Research/ts_compression/l2c/results/results_decompression_l_hier_' +FILENAME.split('.')[0] + '.txt','a') as f:
			#f.write('[')
			[f.write(str(bresults[k]['decompression_latency'])+',') if k != 'apca' else f.write(str(bresults[k]['compressed_ratio'])) for k in bresults]
			#f.write(']')
			#f.write(',')
			f.close()
"""
#compression curves
plt.figure()
results = {}
for error_thresh in range(7,0,-1):
	BASELINES = initialize(ERROR_THRESH=10**(-error_thresh))
	output = run(BASELINES, N=SIZE_LIMIT)
	for k in output:

		if not k in results:
			results[k] = [None]*7

		results[k][ (7-error_thresh) ] = output[k]


ax = plt.axes()
ax.set_prop_cycle('color',[plt.cm.tab20(i) for i in np.linspace(0, 1, len(results))])

for technique in results:
	rgb = np.random.rand(3,)
	plt.plot([v['compressed_ratio'] for v in results[technique]],'s-')

plt.legend([technique for technique in results])
plt.xticks(list(range(0,7)),[ "1e-"+str(r) for r in range(7,0,-1)])
plt.xlim([0,7])
plt.xlabel('Error Threshold %')
plt.title(FILENAME.split('.')[0] + ": Error Dependence" )
plt.ylabel('Compression Ratio')
plt.savefig('thresh_v_ratio' + FILENAME.split('.')[0]+'.png')
"""