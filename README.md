
## Datasets

* [RoadAnomaly21](https://uni-wuppertal.sciebo.de/s/TVR7VxukVrV7fUH/download)
* [RoadObstacle21](https://uni-wuppertal.sciebo.de/s/wQQq2saipS339QA/download)
* [Fishyscapes LostAndFound](https://fishyscapes.com/) - Anomalies
* [LostAndFound](http://wwwlehre.dhbw-stuttgart.de/~sgehrig/lostAndFoundDataset/index.html) - Obstacles

## Evaluation procedure

### Inference

* Place the datasets in `./datasets` (or override with env var `DIR_DATASETS`)
	* `dataset_ObstacleTrack`
	* `dataset_AnomalyTrack`
	* `dataset_LostAndFound` (or provide location in env `DIR_LAF`)
	* `dataset_FishyLAF` (or provide location in env `DIR_FISHY_LAF`)

* Run inference and store results in files. Run inference for the following splits, as the other splits are subsets of those:
	* `AnomalyTrack-all`
	* `ObstacleTrack-all`
	* `LostAndFound-test`
	* `LostAndFound-train`

```python
import numpy as np
from tqdm import tqdm
import cv2 as cv
from road_anomaly_benchmark.evaluation import Evaluation

def method_dummy(image, **_):
	""" Very naive method: return color saturation """
	image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV_FULL)
	anomaly_p = image_hsv[:, :, 1].astype(np.float32) * (1./255.)
	return anomaly_p


def main():

	ev = Evaluation(
		method_name = 'Dummy', 
		dataset_name = 'ObstacleTrack-all',
		# dataset_name = 'AnomalyTrack-test',
	)

	for frame in tqdm(ev.get_frames()):
		# run method here
		result = method_dummy(frame.image)
		# provide the output for saving
		ev.save_output(frame, result)

	# wait for the background threads which are saving
	ev.wait_to_finish_saving()
```

The files will be stored in `./outputs/anomaly_p/...`. The storage directory can be overriden with env var `DIR_OUTPUTS`.
There are also some methods already implemented and available in `some_methods_inference_public.py`. 

**Tip**: to set environment variables `export ENV_VAR=VALUE`.

### Metrics

This step will also create plots in `./outputs/{metric}` (override with env var `DIR_OUTPUTS`)

* Metrics for anomaly track, splits *AnomalyTrack-validation, FishyLAFAnomaly-val*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods AnomalyTrack-validation,FishyLAFAnomaly-val
python -m road_anomaly_benchmark metric SegEval-AnomalyTrack $methods AnomalyTrack-validation,FishyLAFAnomaly-val
```

* Metrics for obstacle track, splits *ObstacleTrack-validation, LostAndFound-testNoKnown*

```bash
methods=Method1,Method2

python -m road_anomaly_benchmark metric PixBinaryClass $methods ObstacleTrack-validation,LostAndFound-testNoKnown
python -m road_anomaly_benchmark metric SegEval-ObstacleTrack $methods ObstacleTrack-validation,LostAndFound-testNoKnown
```

#### Visualization

The anomaly scores can be visualized without ground truth by running with `--only-frame-vis` flag.

```bash
python -m road_anomaly_benchmark metric PixBinaryClass $methods $dsets --only-frame-vis
```

If ground truths are available, the `--frame-vis` option both evaluates the metric and generates visualizations with the ROI region marked. The flag could be used for LostAndFound for example:

```bash
python -m road_anomaly_benchmark metric PixBinaryClass $methods LostAndFound-testNoKnown --frame-vis
```

### Plots and Tables

The following code snippets for comparisons of methods with plots and tables can only be used if ground truths are available. The general syntax for comparing methods is as follows:

```bash
python -m road_anomaly_benchmark comparison MyComparison metric1,metric2 method1,method2 dset1,dset2
```

* Anomaly splits: *AnomalyTrack-validation, FishyLAFAnomaly-val*

```bash
# Anomaly track tables
python -m road_anomaly_benchmark comparison TableAnomaly1 PixBinaryClass,SegEval-AnomalyTrack $methods_ano AnomalyTrack-validation --names names.json
python -m road_anomaly_benchmark comparison TableAnomaly2 PixBinaryClass,SegEval-AnomalyTrack $methods_ano FishyLAFAnomaly-val --names names.json
```

* Obstacle splits: *ObstacleTrack-validation, LostAndFound-testNoKnown*

```bash
# Obstacle track tables
python -m road_anomaly_benchmark comparison TableObstacle1 PixBinaryClass,SegEval-ObstacleTrack $methods_obs ObstacleTrack-validation --names names.json
python -m road_anomaly_benchmark comparison TableObstacle2 PixBinaryClass,SegEval-ObstacleTrack $methods_obs LostAndFound-testNoKnown --names names.json
```

## Citation
If you use this repository, please consider citing our [paper](https://arxiv.org/abs/2104.14812):
	
	@inproceedings{chan2021segmentmeifyoucan,
		title={SegmentMeIfYouCan: A Benchmark for Anomaly Segmentation},
		author={Robin Chan and Krzysztof Lis and Svenja Uhlemeyer and Hermann Blum and Sina Honari and Roland Siegwart and Pascal Fua and Mathieu Salzmann and Matthias Rottmann},
		booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
		year={2021},
	}