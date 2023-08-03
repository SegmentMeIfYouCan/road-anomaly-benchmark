
import logging
log = logging.getLogger('road_anomaly_benchmark.__main__')
import json
from pathlib import Path

import click
import numpy as np

from .paths import DIR_OUTPUTS
from .evaluation import Evaluation
from .metrics import MetricRegistry

def name_list(name_list):
	return [name for name in name_list.split(',') if name]

@click.group()
def main():
	...

@main.command()
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
@click.option('--limit-length', type=int, default=0)
@click.option('--parallel/--no-parallel', default=True)
# @click.option('--frame-vis/--no-frame-vis', default=False)
@click.option('--frame-vis', 'frame_vis', flag_value="True")
@click.option('--no-frame-vis', 'frame_vis', flag_value="False")
@click.option('--only-frame-vis', 'frame_vis', flag_value="only")
@click.option('--default-instancer/--own-instancer', default=True)
def metric(method_names, metric_names, dataset_names, limit_length, parallel, frame_vis, default_instancer):

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	for dset in dataset_names:
		for method in method_names:
			for metric in metric_names:

				log.info(f'Metric: {metric} | Method : {method} | Dataset : {dset}')

				ev = Evaluation(
					method_name = method, 
					dataset_name = dset,
				)

				ag = ev.calculate_metric_from_saved_outputs(
					metric,
					sample = (dset, range(limit_length)) if limit_length != 0 else None,
					parallel = parallel,
					show_plot = False,
					frame_vis = frame_vis,
					default_instancer = default_instancer,
				)

			


COMPARISON_HTML_TEMPLATE = """
<html>
<head>
	<meta charset="utf-8">
	<meta http-equiv="content-type" content="text/html; charset=utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	<title> {title} </title>
	<link rel="stylesheet" href="https://cdn.datatables.net/1.11.3/css/jquery.dataTables.min.css">
</head>
<body>
	{table}
	<script src="https://code.jquery.com/jquery-3.6.0.slim.min.js" integrity="sha256-u7e5khyithlIdTpu22PHhENmPcRdFiHRjhAuHcs05RI=" crossorigin="anonymous"></script>
	<script src="https://cdn.datatables.net/1.11.3/js/jquery.dataTables.min.js"></script>
	<script>{script_src}</script> 
</body>
</html>
"""

COMPARISON_HTML_SCRIPT = """
"use strict";
$(document).ready(() => {
	$('table').DataTable({
		paging: false,
	})
})
"""

def wrap_html_table(table, title='Comparisong'):
	# script has { brackets which don't play well with format
	return COMPARISON_HTML_TEMPLATE.format(
		table = table,
		title = title,
		script_src = COMPARISON_HTML_SCRIPT,
	)


def load_leaderboard_scores(json_file_path, dataset_names):
	from pandas import DataFrame

	json_text = Path(json_file_path).read_text()
	
	# Undo the . to - replacements done in `comparison` below
	for metric_name in ['PixBinaryClass', 'SegEval-ObstacleTrack', 'SegEval-AnomalyTrack']:
		json_text = json_text.replace(f'-{metric_name}-', f'.{metric_name}.')
	
	leaderboard = json.loads(json_text)

	rows_by_methodname = {}

	if isinstance(leaderboard, dict):
		# Top level categories in leaderboard.json are organized by datasets
		
		dset_to_category = {
			'ObstacleTrack-test': 'obstacle_track',
			'AnomalyTrack-test': 'anomaly_track',
			'LostAndFound-testNoKnown': 'laf_no_known',
		}
		
		for dset_name in dataset_names:
			entries_for_dset = leaderboard[dset_to_category[dset_name]]

			for entry in entries_for_dset:
				name = entry['method']
				rows_by_methodname.setdefault(name, {}).update(entry)

	return DataFrame.from_records(list(rows_by_methodname.values()), index='method')


def load_scores(method_names, metric_names, dataset_names, load_leaderboard=None):
	from pandas import DataFrame, concat, Series

	columns = {}
	for dset in dataset_names:
		for metric_name in metric_names:
			metric = MetricRegistry.get(metric_name)
	
			ags = []
			for method in method_names:
				try:
					ag = metric.load(method_name = method, dataset_name = dset)
					ags.append(ag)

					for field, val in metric.extracts_fields_for_table(ag).items():
						col_name = f'{dset}.{metric_name}.{field}'
						col = columns.setdefault(
							col_name, 
							Series(dtype=np.float64)
						)
						col[method] = val

				except FileNotFoundError:
					...

				# if "PixBinaryClass" in metric_name:
				# 	metric.plot_many(
				# 		list(ags),
				# 		f'{comparison_name}_{dset}',
				# 		method_names = rename_methods,
				# 		plot_formats = plot_formats,
				# 	)

	table = DataFrame(data = columns)

	# Load and append rows from leaderboard.json
	if load_leaderboard:
		table_leaderboard = load_leaderboard_scores(load_leaderboard, dataset_names)
		table = concat([table, table_leaderboard])
	
	return table



@main.command()
@click.argument('comparison_name', type=str)
@click.argument('metric_names', type=str)
@click.argument('method_names', type=str)
@click.argument('dataset_names', type=str)
@click.option('--order-by', type=str, default=None)
@click.option('--names', type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option('--leaderboard', 
	type=click.Path(exists=True, file_okay=True, dir_okay=False), 
	help="Path to leaderboard.json, from which extra methods can be loaded.",
)
# @click.option('--leaderboard-drop-columns', type=str, default="requires_ood,paper,code")
def comparison(comparison_name, method_names, metric_names, dataset_names, order_by=None, names=None, leaderboard=None, plot_formats=None):
	import warnings
	warnings.simplefilter(action='ignore', category=FutureWarning)
	from pandas import DataFrame, Series, concat

	method_names = name_list(method_names)
	metric_names = name_list(metric_names)
	dataset_names = name_list(dataset_names)

	

	# Load metrics
	table = load_scores(method_names, metric_names, dataset_names, load_leaderboard=leaderboard)

	# Order
	order_by = order_by or f'{dataset_names[0]}.{metric_names[0]}.area_PRC'
	if order_by in table:
		table = table.sort_values(order_by, ascending=False)
	else:
		log.warn(f'Order by: no column {order_by}')

	# Renames
	if names is not None:
		rename_map = json.loads(Path(names).read_text())
		rename_dsets = rename_map.get('datasets', {})
		rename_metrics = rename_map.get('metrics', {})

		rename_methods = rename_map.get('methods', {})
		methods_drop = [m1 for (m1, m2) in rename_methods.items() if not m2]
		methods_renames = {m1: m2 for (m1, m2) in rename_methods.items() if m2}

		plot_formats = rename_map.get('plots', {})

		# order_by = rename_metrics.get(order_by, order_by)

		col_renames = {}
		col_removes = []

		for col_name in table.columns:
			name_parts = col_name.split('.', maxsplit=1)
			if name_parts.__len__() == 2: # and name_parts[0] in datasets:
				dset, metric = name_parts
			
				dset = rename_dsets.get(dset, dset)
				metric = rename_metrics.get(metric, metric)
				
				if metric:
					col_renames[col_name] = f'{dset}.{metric}'
				else:
					col_removes.append(col_name)
		
		colorder = rename_map.get('colorder', [])

		# Drop these columns unless specifically requested
		for c in ['requires_ood', 'paper', 'code']:
			if not (c in col_renames or c in colorder):
				col_removes.append(c)
		# print('rename keys', col_renames.keys())

		table = table.drop(columns=col_removes, index=methods_drop, errors='ignore')
		table = table.rename(columns=col_renames, index=methods_renames, errors='ignore')
		
		# column reorder
		print(table.columns)
		if colorder:
			print('Reorder', table.columns, colorder)
			table = table[colorder]


	print(table)

	str_formats = dict(
		float_format = lambda f: f'{100*f:.01f}',
		na_rep = '-',
	)
	table_tex = table.to_latex(**str_formats)
	table_html = wrap_html_table(
		table = table.to_html(
			classes = ('display', 'compact'), 
			**str_formats,
		),
		title = comparison_name,
	)

	# json dump for website
	table['method'] = table.index
	table_json = table.to_json(orient='records')
	table_data = json.loads(table_json)
	table_data = [{k.replace('.', '-'): v for k, v in r.items()} for r in table_data]
	table_json = json.dumps(table_data)

	out_f = DIR_OUTPUTS / 'tables' / comparison_name
	out_f.parent.mkdir(parents=True, exist_ok=True)
	out_f.with_suffix('.html').write_text(table_html)
	out_f.with_suffix('.tex').write_text(table_tex)
	out_f.with_suffix('.json').write_text(table_json)



@main.command()
@click.argument('metric_name', type=str)
@click.argument('method_name', type=str)
@click.argument('dset', type=str)
def read_metric(metric_name, method_name, dset):

	metric = MetricRegistry.get(metric_name)
	res = metric.load(method_name = method_name, dataset_name = dset)
	print(res)
	

if __name__ == '__main__':
	main()
