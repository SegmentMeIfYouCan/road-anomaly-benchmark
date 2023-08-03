
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
		dataset_name = 'ObstacleTrack-validation',
		# dataset_name = 'AnomalyTrack-test',
	)

	for frame in tqdm(ev.get_frames()):
		# run method here
		result = method_dummy(frame.image)
		# provide the output for saving
		ev.save_output(frame, result)

	# wait for the background threads which are saving
	ev.wait_to_finish_saving()

	ev.calculate_metric_from_saved_outputs(
		'PixBinaryClass',
		frame_vis=True,
	)

	ev.calculate_metric_from_saved_outputs(
		'SegEval-ObstacleTrack',
		frame_vis=True,
	)

if __name__ == '__main__':
	main()
