
import dataclasses
from typing import List, Dict
from operator import attrgetter

import numpy as np

from matplotlib import pyplot
from easydict import EasyDict
from pandas import DataFrame, Series

from ..datasets.dataset_io import hdf5_write_hierarchy_to_file, hdf5_read_hierarchy_from_file

@dataclasses.dataclass
class BinaryClassificationCurve:
	method_name : str
	dataset_name : str
#	display_name : str = ''

	area_PRC : float
	curve_recall : np.ndarray
	curve_precision : np.ndarray

	area_ROC : float
	curve_tpr : np.ndarray
	curve_fpr : np.ndarray

	thresholds : np.ndarray = None

	tpr95_fpr : float = -1
	tpr95_threshold : float = -1

	IOU_at_05: float = float('nan')
	PDR_at_05: float = float('nan')

	recall50_threshold : float = -1

	best_f1 : float = -1
	best_f1_threshold : float = -1

	def __iter__(self):
		return dataclasses.asdict(self).items()

	def save(self, path):
		hdf5_write_hierarchy_to_file(path, dataclasses.asdict(self))

	@classmethod
	def from_file(cls, path):
		return cls(**hdf5_read_hierarchy_from_file(path))


def curves_from_cmats(cmats, thresholds, debug_thresholds=False):
	
	# The threshold goes from high to low
	
	# The thresholds are independent for each frame which can result in repeated thresholds.
	# Here we collapse those segments of the curve that would be caused by repeated thresholds.
	# We choose the indices where each threshold appears for the last time.
	# This means that all the samples for this threshold have been accounted for. 

	# Thanks to Jan Ackermann from ETHZ for reporting problems with the AP curves.

	thr_unique_indices = np.concatenate([
		[0],
		np.nonzero(thresholds[:-1] != thresholds[1:])[0],
		[cmats.__len__()-1],
	])	
	# print(f'Reducing sample numbers from {len(thresholds)} thresholds and {len(cmats)} cmats to {len(thr_unique_indices)}')

	# Ignore the entries for repeated thresholds.
	cmats = cmats[thr_unique_indices]
	thresholds = thresholds[thr_unique_indices]

	tp = cmats[:, 0, 0]
	fp = cmats[:, 0, 1]
	fn = cmats[:, 1, 0]
	tn = cmats[:, 1, 1]

	num_pos = tp[0] + fn[0]
	num_neg = fp[0] + tn[0]

	# Ensure the end points of the curve, PR curve starts at precision = 1, recall = 0
	if tp[0] != 0 or fp[0] != 0:
		tp = np.concatenate([[0], tp])
		fp = np.concatenate([[0], fp])
		fn = np.concatenate([[num_pos], fn])
		tn = np.concatenate([[num_neg], tn])
		thresholds = np.concatenate([[float(np.max(thresholds)) + 1], thresholds])

	if tp[-1] != num_pos or fp[-1] != num_neg:
		tp = np.concatenate([tp, [num_pos]])
		fp = np.concatenate([fp, [num_neg]])
		fn = np.concatenate([fn, [0]])
		tn = np.concatenate([tn, [0]])
		thresholds = np.concatenate([thresholds, [float(np.max(thresholds)) - 1]])

	num_steps = tp.shape[0]

	tp_rates = tp / (tp+fn)
	fp_rates = fp / (fp+tn)

	# At a particularly high threshold (at the beginning of the confusion matrix collection)
	# there may not yet be any predictions (tp+fp == 0).
	# Calculating precision in that range would result in a division by zero,
	# we set the precision in that range to 1 to follow how the curve should start.

	n_preds = tp+fp
	precisions = np.ones(num_steps, np.float64)
	precisions[n_preds > 0] = tp[n_preds > 0] / n_preds[n_preds > 0]

	recalls = tp / (tp+fn)

	# if not ( precisions[0] == 1 and recalls[0] == 0 ):
	# 	precisions = np.concatenate([[1], precisions])
	# 	recalls = np.concatenate([[0], recalls])
	# # and ends at recall = 1
	# if not ( recalls[-1] == 1 ):
	# 	precisions = np.concatenate([precisions, [0]])
	# 	recalls = np.concatenate([recalls, [1]])


	f1_scores = (2 * tp) / (2 * tp + fp + fn)

	tpr95_index = np.searchsorted(tp_rates, 0.95)
	if tpr95_index < tp_rates.shape[0]:
		fpr_tpr95 = fp_rates[tpr95_index]
		tpr95_threshold = float(thresholds[tpr95_index])
	else:
		# tpr95 was not reached
		fpr_tpr95 = 1.0
		tpr95_threshold = 0.0

	recall50_index = np.searchsorted(recalls, 0.50)
	recall50_threshold = float(thresholds[recall50_index])

	ix = np.nanargmax(f1_scores)

	if debug_thresholds:
		print('f1_scores', 'len', len(f1_scores), f1_scores)
		print('thresholds', 'len', len(thresholds), thresholds)

		from matplotlib import pyplot
		fig, plot = pyplot.subplots(1, 1)
		plot.plot(f1_scores, label='f1 scores')
		plot.plot(thresholds, label='thresholds')
		fig.legend()
		fig.savefig('debug_best_threshold.png')



	best_f1_threshold = float(thresholds[ix])
	best_f1 = f1_scores[ix]

	print(
		'ap-sum', np.sum(np.diff(recalls) * precisions[1:]),
		'ap-trapz', np.trapz(precisions, recalls),
	)

	# Change the sum strategy to match sklearn rather than 
	# taking the triangular area under the rising curve.
	ap = np.sum(np.diff(recalls) * precisions[1:])

	return EasyDict(
		# curves
		curve_tpr = tp_rates,
		curve_fpr = fp_rates,
		curve_precision = precisions,
		curve_recall = recalls,
		
		thresholds = thresholds,

		# areas
		area_ROC = np.trapz(tp_rates, fp_rates),
		area_PRC = ap,

		tpr95_fpr = fpr_tpr95,
		tpr95_threshold = tpr95_threshold,

		recall50_threshold = recall50_threshold,
		best_f1_threshold = best_f1_threshold,
		best_f1 = best_f1

	)


def select_points_for_curve(x, y, num_points, value_range=(0, 1)):
	"""
	x is ascending
	"""

	range_start, range_end = value_range
	thresholds = np.linspace(range_start, range_end, num=num_points-2, dtype=np.float64)

	indices = []

	# points spaced equally in x space
	idx = 0
	for thr in thresholds:
		# binary search for the next threshold
		ofs= np.searchsorted(x[idx:], thr)
		# print(f'{idx}/{x.size} + {ofs} thr {thr}')
		idx += ofs
		indices.append(idx)


	# points spaced equally as percentiles
	if x.size > num_points:
		indices += list(range(0, x.size, x.size // num_points))
	else:
		indices = list(range(x.size))

	# first and last point is always included
	indices.append(0)
	indices.append(x.size-1)

	# sort and remove duplicated
	indices = np.unique(indices)

	# print(indices, 'vs', x.size)

	if indices[-1] == x.size:
		indices = indices[:-1]

	# print(indices)

	return dict(
		indices = indices,
		curve_x = x[indices],
		curve_y = y[indices],
	)


def reduce_curve_resolution(cinfo : BinaryClassificationCurve, num_pts : int = 128) -> BinaryClassificationCurve:
	"""
	Reduces curve resolution for plotting
	"""

	if cinfo.curve_precision.__len__() <= num_pts:
		return cinfo

	prc = select_points_for_curve(
		cinfo.curve_recall, 
		cinfo.curve_precision, 
		num_points = num_pts,
	)

	indices = prc['indices']

	# roc = select_points_for_curve(
	# 	curve_info.curve_fpr, 
	# 	curve_info.curve_tpr, 
	# 	num_points = num_pts,
	# )
	
	return dataclasses.replace(
		cinfo,
		curve_recall = prc['curve_x'],
		curve_precision = prc['curve_y'],
		# curve_fpr = roc['curve_x']
		# curve_tpr = roc['curve_y']
		curve_fpr = cinfo.curve_fpr[indices],
		curve_tpr = cinfo.curve_tpr[indices],
		thresholds = cinfo.thresholds[indices],
	)


def plot_classification_curves_draw_entry(plot_roc : pyplot.Axes, plot_prc : pyplot.Axes, curve_info : BinaryClassificationCurve, display_name : str, format=None):

	fmt_args = {}

	if format is not None:
		segs = format.split()

		if segs.__len__() >= 1:
			fmt_args['color'] = segs[0]

		if segs.__len__() >= 2:
			fmt_args['linestyle'] = segs[1]

		if segs.__len__() >= 3:
			fmt_args['marker'] = segs[2]

	if plot_prc is not None:
		curves = select_points_for_curve(
			curve_info.curve_recall, 
			curve_info.curve_precision, 
			num_points = 256,
		)
		curve_recall = curves['curve_x']
		curve_precision = curves['curve_y']
		# curve_recall = ci_red.curve_recall
		# curve_precision = ci_red.curve_precision

		plot_prc.plot(curve_recall, curve_precision,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{curve_info.area_PRC:.02f} {display_name}',
			#marker = '.',
			**fmt_args,
		)

	if plot_roc is not None:
		curves = select_points_for_curve(
			curve_info.curve_fpr, 
			curve_info.curve_tpr, 
			num_points = 256,
		)
		curve_fpr = curves['curve_x']
		curve_tpr = curves['curve_y']

		plot_roc.plot(curve_fpr, curve_tpr,
			#fmt,
			# label='{lab:<24}{a:.02f}'.format(lab=label, a=area_under),
			label=f'{curve_info.area_ROC:.03f} {display_name}',
			# marker = '.',
			**fmt_args,
		)


def plot_classification_curves(curve_infos : List[BinaryClassificationCurve], method_names : Dict[str, str] = {}, plot_formats = {}, plot_best = None):
	
	table_scores = DataFrame(data = [
		Series({
			'AveragePrecision': crv.area_PRC, 
			'AUROC': crv.area_ROC, 
			'FPR-at-95-TPR:': crv.tpr95_fpr,
			'IOU': crv.IOU_at_05,
			'PDR': crv.PDR_at_05,
			},
			name = method_names.get(crv.method_name, crv.method_name),
		)
		for crv in curve_infos
	])
	table_scores = table_scores.sort_values('AveragePrecision', ascending=False)

	try:
		fig = pyplot.figure(figsize=(18, 8))
	except AttributeError:
		# Can cause a failure if trying to load GTK
		import matplotlib
		matplotlib.use('agg')
		fig = pyplot.figure(figsize=(18, 8))

	plot_roc, plot_prc = fig.subplots(1, 2)
	
	plot_prc.set_xlabel('recall')
	plot_prc.set_ylabel('precision')
	plot_prc.set_xlim([0, 1])
	plot_prc.set_ylim([0, 1])

	plot_roc.set_xlabel('false positive rate')
	plot_roc.set_ylabel('true positive rate')
	plot_roc.set_xlim([0, 0.2])

	# sort descending by AP
	curve_infos_sorted = list(curve_infos)
	curve_infos_sorted.sort(key=attrgetter('area_PRC'), reverse=True)

	for k, crv in enumerate(curve_infos_sorted):
		plot_classification_curves_draw_entry(
			plot_prc = plot_prc, 
			plot_roc = plot_roc, 
			curve_info = crv,
			display_name = method_names.get(crv.method_name, crv.method_name),
			format = plot_formats.get(crv.method_name),
		)
		if k == plot_best:
			break

	def make_legend(plot_obj, position='lower right', title=None):
		handles, labels = plot_obj.get_legend_handles_labels()
		plot_obj.legend(handles, labels, loc=position, title=title)

		plot_obj.grid(True)

	make_legend(plot_prc, 'lower left', title='Average Precision')
	make_legend(plot_roc, 'lower right', title='AUROC')

	fig.tight_layout()

	return EasyDict(
		plot_figure = fig,
		score_table = table_scores,
	)



