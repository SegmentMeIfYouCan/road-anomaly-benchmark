
from typing import List#, Literal
from pathlib import Path

import numpy as np
from matplotlib import pyplot
from easydict import EasyDict

from .base import EvaluationMetric, MetricRegistry, save_figure, save_table
from .pixel_classification_curves import BinaryClassificationCurve, curves_from_cmats, plot_classification_curves, reduce_curve_resolution

from ..evaluation import DIR_OUTPUTS
from ..jupyter_show_image import adapt_img_data, get_heat, imwrite


def binary_confusion_matrix(
		prob : np.ndarray, gt_label_bool : np.ndarray, 
		num_bins : int = 1024, bin_strategy = 'uniform', # : Literal['uniform', 'percentiles'] = 'uniform',
		normalize : bool = False, dtype = np.float64):
	
	area = gt_label_bool.__len__()

	gt_area_true = np.count_nonzero(gt_label_bool)
	gt_area_false = area - gt_area_true

	prob_at_true = prob[gt_label_bool]
	prob_at_false = prob[~gt_label_bool]

	if bin_strategy == 'uniform':
		# bins spread uniforms in 0 .. 1
		bins = num_bins
		histogram_range = [-0.01, 1.01]

	elif bin_strategy == 'percentiles':
		# dynamic bins representing the range of occurring values
		# bin edges are following the distribution of positive and negative pixels

		# choose thresholds to surround the range of values
		vmin = np.min(prob)
		vmax = np.max(prob)
		vrange = vmax-vmin
		eps = np.maximum(vrange*1e-2, 1e-2) # make sure there is some separation between the thresholds

		bins = [
			[vmin - eps, vmax + eps]
		]

		if prob_at_true.size:
			bins += [
				np.quantile(prob_at_true, np.linspace(0, 1, min(num_bins//2, prob_at_true.size))),
			]
		if prob_at_false.size:
			bins += [
				np.quantile(prob_at_false, np.linspace(0, 1, min(num_bins//2, prob_at_false.size))),
			]

			
		bins = np.concatenate(bins)
		
		# sort and remove duplicates, duplicated cause an exception in np.histogram
		bins = np.unique(bins)
		

		histogram_range = None

	# the area of positive pixels is divided into
	#	- true positives - above threshold
	#	- false negatives - below threshold
	tp_rel, _ = np.histogram(prob_at_true, bins=bins, range=histogram_range)
	# the curve goes from higher thresholds to lower thresholds
	tp_rel = tp_rel[::-1]
	# cumsum to get number of tp at given threshold
	tp = np.cumsum(tp_rel)
	# GT-positives which are not TP are instead FN
	fn = gt_area_true - tp

	# the area of negative pixels is divided into
	#	- false positives - above threshold
	#	- true negatives - below threshold
	fp_rel, bin_edges = np.histogram(prob_at_false, bins=bins, range=histogram_range)
	# the curve goes from higher thresholds to lower thresholds
	bin_edges = bin_edges[::-1]
	fp_rel = fp_rel[::-1]
	# cumsum to get number of fp at given threshold
	fp = np.cumsum(fp_rel)
	# GT-negatives which are not FP are instead TN
	tn = gt_area_false - fp

	cmat_sum = np.array([
		[tp, fp],
		[fn, tn],
	]).transpose(2, 0, 1).astype(dtype)

	# cmat_rel = np.array([
	# 	[tp_rel, fp_rel],
	# 	[-tp_rel, -fp_rel],
	# ]).transpose(2, 0, 1).astype(dtype)
	
	if normalize:
		cmat_sum *= (1./area)
		# cmat_rel *= (1./area)

	return EasyDict(
		bin_edges = bin_edges,
		cmat_sum = cmat_sum,
		# cmat_rel = cmat_rel,
		tp_rel = tp_rel,
		fp_rel = fp_rel,
		num_pos = gt_area_true,
		num_neg = gt_area_false,
	)


def test_binary_confusion_matrix():
	pp = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
	gt = np.array([0, 0, 0, 0, 0, 1, 1, 0, 1], dtype=bool)
	cmat = binary_confusion_matrix(pp, gt, levels=20).cmat_sum
	cmat_all_p = np.sum(cmat[:, :, 0], axis=1)
	cmat_all_n = np.sum(cmat[:, :, 1], axis=1)

	print(cmat_all_p, cmat_all_n)

	pyplot.plot(cmat[:, 0, 1] / cmat_all_n, cmat[:, 0, 0] / cmat_all_p)




@MetricRegistry.register_class()
class MetricPixelClassification(EvaluationMetric):

	configs = [
		EasyDict(
			name = 'PixBinaryClass-uniThr',
			# pixel scores in a given image are quantized into bins, 
			# so that big datasets can be stored in memory and processed in parallel
			num_bins = 4096,
			bin_strategy = 'uniform',
		),
		EasyDict(
			name = 'PixBinaryClass',
			num_bins = 768,
			bin_strategy = 'percentiles',
		)
	]

	@property
	def name(self):
		return self.cfg.name
	
	@staticmethod
	def vis_frame(fid, dset_name, method_name, mask_roi, anomaly_p, image = None, label_pixel_gt = None, **_):
		h, w = mask_roi.shape[:2]

		canvas = image.copy() if image is not None else np.zeros((h, w, 3), dtype=np.uint8)
		heatmap_color = adapt_img_data(anomaly_p)
		canvas[mask_roi] = canvas[mask_roi]//2 + heatmap_color[mask_roi]//2
		imwrite(
			DIR_OUTPUTS / f'vis_PixelClassification' / method_name / dset_name / f'{fid}_demo_anomalyP.webp',
			canvas,
		)

		anomaly_heat = get_heat(anomaly_p, overlay=label_pixel_gt)
		imwrite(
			DIR_OUTPUTS / f'vis_PixelClassification' / method_name / dset_name / f'{fid}_demo_anomalyP_heat.webp',
			anomaly_heat,
		)


	def process_frame(self, label_pixel_gt : np.ndarray, anomaly_p : np.ndarray, fid : str=None, dset_name : str=None, method_name : str=None, visualize : bool = True, **_):
		"""
		@param label_pixel_gt: HxW uint8
			0 = road
			1 = obstacle
			255 = ignore
		@param anomaly_p: HxW float16
			heatmap of per-pixel anomaly detection, value from 0 to 1
		@param fid: frame identifier, for saving extra outputs
		@param dset_name: dataset identifier, for saving extra outputs
		"""
		try:
			mask_roi = label_pixel_gt < 255
		except TypeError:
			raise RuntimeError(f"No ground truth available for {fid}. Please check dataset path...")
		labels_in_roi = label_pixel_gt[mask_roi]
		predictions_in_roi = anomaly_p[mask_roi]

		bc = binary_confusion_matrix(
			prob = predictions_in_roi,
			gt_label_bool = labels_in_roi.astype(bool),
			num_bins = self.cfg.num_bins,
			bin_strategy = self.cfg.bin_strategy,
		)
		# bc.cmats, bc.bins

		# visualization
		if visualize and fid is not None and dset_name is not None and method_name is not None:
			self.vis_frame(fid=fid, dset_name=dset_name, method_name=method_name, mask_roi=mask_roi,
						   anomaly_p=anomaly_p, label_pixel_gt=label_pixel_gt, **_)

		#print('Vrange', np.min(predictions_in_roi), np.mean(predictions_in_roi), np.max(predictions_in_roi))

		return bc


	@staticmethod
	def aggregate_fixed_bins(frame_results):
		
		# bin edges are the same in every frame
		bin_edges = frame_results[0].bin_edges
		thresholds = bin_edges[1:]

		# each frame has the same thresholds, so we can sum the cmats
		cmat_sum = np.sum([result.cmat_sum for result in frame_results], axis=0)

		return EasyDict(
			cmat = cmat_sum,
			thresholds = thresholds,
		)

	@staticmethod
	def aggregate_dynamic_bins(frame_results):

		thresholds = np.concatenate([r.bin_edges[1:] for r in frame_results])

		tp_relative = np.concatenate([r.tp_rel for r in frame_results], axis=0)
		fp_relative = np.concatenate([r.fp_rel for r in frame_results], axis=0)

		num_positives = sum(r.num_pos for r in frame_results)
		num_negatives = sum(r.num_neg for r in frame_results)


		threshold_order = np.argsort(thresholds)[::-1]

		# We start at threshold = 1, and lower it
		# Initially, prediction=0, all GT=1 pixels are false-negatives, and all GT=0 pixels are true-negatives.

		tp_cumu = np.cumsum(tp_relative[threshold_order].astype(np.float64))
		fp_cumu = np.cumsum(fp_relative[threshold_order].astype(np.float64))

		cmats = np.array([
			# tp, fp
			[tp_cumu, fp_cumu],
			# fn, tn
			[num_positives - tp_cumu, num_negatives - fp_cumu],
		]).transpose([2, 0, 1])

		return EasyDict(
			cmat = cmats,
			thresholds = thresholds[threshold_order],
		)

	def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
		# fuse cmats FIXED BINS

		if self.cfg.bin_strategy == 'uniform':
			ag = self.aggregate_fixed_bins(frame_results)
		else:
			ag = self.aggregate_dynamic_bins(frame_results)

		thresholds = ag.thresholds
		cmats = ag.cmat

		curves = curves_from_cmats(cmats, thresholds)

		bc_info = BinaryClassificationCurve(
			method_name = method_name,
			dataset_name = dataset_name,
			**curves,
		)

		return bc_info

	def persistence_path_data(self, method_name, dataset_name):
		return DIR_OUTPUTS / self.name / 'data' / f'PixClassCurve_{method_name}_{dataset_name}.hdf5'

	def persistence_path_plot(self, comparison_name, plot_name):
		return DIR_OUTPUTS / self.name / 'plot' / f'{comparison_name}__{plot_name}'

	def save(self, aggregated_result, method_name : str, dataset_name : str, path_override : Path = None):
		out_path = path_override or self.persistence_path_data(method_name, dataset_name)
		aggregated_result.save(out_path)

		c_lowres = reduce_curve_resolution(aggregated_result, num_pts=256)
		c_lowres.save(out_path.with_name(out_path.name.replace('PixClassCurve', 'PixClassCurve-simplified')))

	def load(self, method_name : str, dataset_name : str, path_override : Path = None):
		out_path = path_override or self.persistence_path_data(method_name, dataset_name)
		out_path_simplified = out_path.with_name(out_path.name.replace('PixClassCurve', 'PixClassCurve-simplified'))

		if out_path.is_file():
			return BinaryClassificationCurve.from_file(out_path)
		elif out_path_simplified.is_file():
			return BinaryClassificationCurve.from_file(out_path_simplified)
		else:
			raise FileNotFoundError(f'No saved curve at {out_path} or {out_path_simplified}')

	def fields_for_table(self):
		return ['area_PRC', 'tpr95_fpr', 'best_f1']

	def plot_many(self, aggregated_results : List, comparison_name : str, close : bool = True, method_names={}, plot_formats={}):

		cinfos = [
			reduce_curve_resolution(cinfo, num_pts=256)
			for cinfo in aggregated_results
		]

		vis_res = plot_classification_curves(cinfos, method_names=method_names, plot_formats=plot_formats)
		fig = vis_res.plot_figure
		table = vis_res.score_table
			
		save_figure(self.persistence_path_plot(comparison_name, 'PixClassCurves'), fig)

		if close:
			pyplot.close(fig)

		save_table(self.persistence_path_plot(comparison_name, 'PixClassTable'), table)
