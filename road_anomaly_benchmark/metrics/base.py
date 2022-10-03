 
from typing import List
from pathlib import Path

import numpy as np
from easydict import EasyDict

from ..datasets.dataset_registry import Registry


MetricRegistry = Registry()


class EvaluationMetric:

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)

	def init(self, method_name, dataset_name):
		...

	def process_frame(self, label_pixel_gt : np.ndarray, anomaly_p : np.ndarray, dset_name : str=None, method_name : str=None, visualize : bool = True, **_):
		"""
		@param frame: dict containing frame fields, such as `image`, `semantic_class_gt`, `pred_anomaly_p`
		@return: 
		"""
		...

	def aggregate(self, frame_results : list, method_name : str, dataset_name : str):
		"""
		@param frame_results: sequence of outputs of `process_frame` for the whole dataset
		@return: 
		"""
		...
	
	def save(self, aggregated_result, method_name : str, dataset_name : str, path_override : Path = None):
		...

	def load(self, method_name : str, dataset_name : str, path_override : Path = None):
		...

	def fields_for_table(self) -> List[str]:
		return []

	def extracts_fields_for_table(self, ag):
		return EasyDict({
			f: getattr(ag, f)
			for f in self.fields_for_table()
		})

	def plot_many(self, aggregated_results : List, comparison_name : str, close : bool = True):
		...

	def plot_single(self, aggregated_result, close : bool = True):
		ag = aggregated_result
		self.plot_many([ag], f'{ag.method_name}_{ag.dataset_name}', close=close)


def save_figure(path, fig):
	path.parent.mkdir(parents=True, exist_ok=True)

	for fmt in ('png', 'svg', 'pdf'):
		fig.savefig(path.with_suffix(f'.{fmt}'))
	

def save_table(path, table):
	path.parent.mkdir(parents=True, exist_ok=True)
	
	path.with_suffix('.html').write_text(table.to_html())
	path.with_suffix('.tex').write_text(table.style.to_latex())


	# table_tex = table_pd.to_latex(
	# 	float_format = float_format,
	# 	columns = [(ds, metric) for (ds, metric) in table_columns.keys() if ds != 'Any']
	# )

	# table_html = table_pd.to_html(
	# 	float_format = float_format,
	# )




