
from pathlib import Path
from os import environ
from operator import itemgetter
import logging, re

from easydict import EasyDict
import numpy as np

from ..paths import DIR_DATASETS
from .dataset_registry import DatasetRegistry
from .dataset_io import DatasetBase, ChannelLoaderImage


log = logging.getLogger(__name__)

class DatasetRA(DatasetBase):

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()

	def discover(self):
		""" Discover frames in file system """
		path_template = Path(self.channels['image'].resolve_template(
			dset = self,
			fid = '*',
		))
		# print(path_template, path_template.parent, path_template.name)
		fids = [p.stem for p in path_template.parent.glob(path_template.name)]
		fids.sort()
		self.set_frames([EasyDict(fid=fid) for fid in fids])
		self.check_size()


	def check_size(self):
		desired_len = self.cfg.get('expected_length')
		actual_len = self.__len__()

		if desired_len is not None and actual_len != desired_len:
			raise ValueError(f'The dataset should have {desired_len} frames but found {actual_len}')

	def get_frame(self, key, *channels):

		channels = set(channels)
		wants_labels_explicitly = False
		if 'label_pixel_gt' in channels:
			wants_labels_explicitly = True
			channels.remove('label_pixel_gt')
			channels.add('semantic_class_gt')

		fr = super().get_frame(key, *channels)

		sem_gt = fr.get('semantic_class_gt')
		if sem_gt is not None:
			h, w = sem_gt.shape[:2]
			label = np.full((h, w), 255, dtype=np.uint8)
			label[sem_gt == self.cfg.classes.usual] = 0

			anomaly = self.cfg.classes.anomaly
			if isinstance(anomaly, (tuple, list)) and anomaly.__len__() == 2:
				range_low, range_high = anomaly
				anomaly_mask = (range_low <= sem_gt) & (sem_gt <= range_high)
			else:
				anomaly_mask = sem_gt == anomaly

			label[anomaly_mask] = 1

			fr['label_pixel_gt'] = label
		elif wants_labels_explicitly:
			raise KeyError(f'No labels for {key} in {self}')
		else:
			fr['label_pixel_gt'] = None


		return fr


@DatasetRegistry.register_class()
class DatasetAnomalyTrack(DatasetRA):

	configs = [
		dict(
			name = 'AnomalyTrack-all',
			dir_root = DIR_DATASETS / 'dataset_AnomalyTrack',
			img_fmt = 'jpg',
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 255,
			),
			expected_length = 110,
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}


@DatasetRegistry.register_class()
class DatasetObstacleTrack(DatasetRA):

	CLASS_IDS = dict(
		road = 0,
		obstacle = 1,
		ignore = 255,

		usual = 0,
		anomaly = 1,
	)

	DEFAULTS = dict(
		dir_root = DIR_DATASETS / 'dataset_ObstacleTrack',
		img_fmt = 'webp',
		classes = CLASS_IDS,
		name_for_persistence = 'ObstacleTrack-all',
	)

	SCENES_ALL = {
		'curvy-street', 'one-way-street', # obstacle track
		'gravel', 'greyasphalt', 'motorway', 'paving', 'darkasphalt', # RO 2020
		'darkasphalt2', # RO 2020 dog
		'snowstorm1', 'snowstorm2', # RO 2021
		'driveway', # night
		'validation',
	}

	# splits for per-scene breakdown
	SCENE_SPLITS = {
		'curvy': ['curvy-street'],
		'darkasphalt': ['darkasphalt'],
		'darkasphaltDog': ['darkasphalt2'], # dog
		'darkasphaltAll': ['darkasphalt', 'darkasphalt2'],
		'gravel': ['gravel'],
		'greyasphalt': ['greyasphalt'],
		'motorway': ['motorway'],
		'shiny': ['one-way-street'], # sun reflects off wet road
		'paving': ['paving'],
		'night': ['driveway'],
		'snowstorm': ['snowstorm1', 'snowstorm2'],
		'validation': ['validation'],
	}

	configs = [
		dict(
			# default: exclude special weather and night
			name = 'ObstacleTrack-test',
			scenes = SCENES_ALL.difference({'snowstorm1', 'snowstorm2', 'driveway', 'validation'}),
			expected_length = 327,
			**DEFAULTS,
		),
		dict(
			# all
			name = 'ObstacleTrack-all',
			scenes = SCENES_ALL,
			# expected_length = 452,
			**DEFAULTS,
		),
		dict(
			# exclude night
			name = 'ObstacleTrack-noNight',
			scenes = SCENES_ALL.difference({'driveway'}),
			expected_length = 382,
			**DEFAULTS,
		),
		dict(
			# night
			name = 'ObstacleTrack-night',
			scenes = {'driveway'},
			expected_length = 30,
			**DEFAULTS,
		),
		dict(
			# night
			name = 'ObstacleTrack-snowstorm',
			scenes = {'snowstorm1', 'snowstorm2'},
			expected_length = 55,
			**DEFAULTS,
		),
		dict(
			# validation
			name='ObstacleTrack-validation',
			scenes={'validation'},
			exclude_prefix={
				'validation_19',
				'validation_21',
				'validation_22',
				'validation_23',
				'validation_24',
				'validation_25',
				'validation_26',
				'validation_27',
				'validation_28',
				'validation_29',
			},
			expected_length=30,
			**DEFAULTS,
		),
	]

	for splitname, scenes in SCENE_SPLITS.items():
		configs.append(dict(
			name = f'ObstacleScene-{splitname}',
			scenes = set(scenes),
			**DEFAULTS,
		))

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}

	def set_frames(self, frame_list):
		""" Filter frames by requested scenes """
		frames_filtered = [
			fr for fr in frame_list
			if fr.fid.split('_')[0] in self.cfg.scenes
		]

		excluded_prefixes = self.cfg.get('exclude_prefix')
		if excluded_prefixes is not None:
			frlen = frames_filtered.__len__()
			frames_filtered = [
				fr for fr in frames_filtered
				if not any([
					fr.fid.startswith(p) for p in excluded_prefixes
				])
			]
			print(f'Exclude {frlen} -> {frames_filtered.__len__()}')

		super().set_frames(frames_filtered)

@DatasetRegistry.register_class()
class DatasetWeather(DatasetRA):

	configs = [
		dict(
			name = 'RoadObstacleWeather-v1',
			dir_root = DIR_DATASETS / 'dataset_RoadObstacleWeather_v1',
			# classes = dict(
			# 	road = 253,
			# 	obstacle = 254,
			# 	ignore = 0,
			# )
		),
		dict(
			name = 'RoadObstacleExtra-v1',
			dir_root = DIR_DATASETS / 'dataset_RoadObstacleExtra',
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.jpg"),
		#'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}

@DatasetRegistry.register_class()
class DatasetLostAndFound(DatasetRA):
	"""
	https://github.com/mcordts/cityscapesScripts#dataset-structure
	"""

	DIR_LAF = Path(environ.get('DIR_LAF', DIR_DATASETS / 'dataset_LostAndFound'))

	LAF_CLASSES = dict(
		ignore = 0,
		usual = 1, # road
		anomaly = [2, 200], # range
	)

	DEFAULTS = dict(
		dir_root = DIR_LAF,
		classes = LAF_CLASSES,
	)

	configs = [
		dict(
			name = 'LostAndFound-train',
			split = 'train',
			expected_length = 1036,
			**DEFAULTS,
		),
		dict(
			name = 'LostAndFound-test',
			split = 'test',
			expected_length = 1203,
			**DEFAULTS,
		),
		dict(
			name = 'LostAndFound-trainValid',
			split = 'train',
			name_for_persistence = 'LostAndFound-train',

			# invalid frames are those where np.count_nonzero(labels_source) is 0
			exclude_frame_indices = [44,  67,  88, 109, 131, 614],
			expected_length = 1030,

			**DEFAULTS,
		),
		dict(
			name = 'LostAndFound-testValid',
			name_for_persistence = 'LostAndFound-test',
			split = 'test',

			# invalid frames are those where np.count_nonzero(labels_source) is 0
			exclude_frame_indices = [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
			expected_length = 1186,

			**DEFAULTS,
		),

		dict(
			# valid test set, excluding known objects - pedestrians and bicycles
			name = 'LostAndFound-testNoKnown',
			name_for_persistence = 'LostAndFound-test',
			split = 'test',

			# invalid frames are those where np.count_nonzero(labels_source) is 0
			exclude_frame_indices = [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793],
			exclude_prefix = {
				'15_Rechbergstr_Deckenpfronn',  # children
    			'01_Hanns_Klemm_Str_45_000006',  # velo
    			'01_Hanns_Klemm_Str_45_000007',  # velo
    			'10_Schlossberg_9_000004',  # velo
			},
			expected_length = 1043,

			**DEFAULTS,
		),
	]

	channels = {
		'image': ChannelLoaderImage(
			'{dset.cfg.dir_root}/leftImg8bit/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_leftImg8bit.{dset.img_fmt}',
		),
		'semantic_class_gt': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtCoarse/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_gtCoarse_labelIds.png',
		),
		# 'semantic_class_gt_tid': ChannelLoaderImage(
		# 	'{dset.cfg.dir_root}/gtCoarse/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_gtCoarse_labelTrainIds.png',
		# ),
		'instances': ChannelLoaderImage(
			'{dset.cfg.dir_root}/gtCoarse/{dset.cfg.split}/{scene_id:02d}_{scene_name}/{fid}_gtCoarse_instanceIds.png',
		),
	}

	RE_LAF_NAME = re.compile(r'([0-9]{2})_(.*)_([0-9]{6})_([0-9]{6})')
	LAF_SUFFIX_LEN = '_leftImg8bit'.__len__()

	@classmethod
	def laf_id_from_image_path(cls, path, **_):
		fid = path.stem[:-cls.LAF_SUFFIX_LEN]

		m = cls.RE_LAF_NAME.match(fid)

		return EasyDict(
			fid = fid,
			scene_id = int(m.group(1)),
			scene_name = m.group(2),
			scene_seq = int(m.group(3)),
			scene_time = int(m.group(4))
		)


	def discover(self):
		img_dir = Path(self.cfg.dir_root) / 'leftImg8bit' / self.cfg.split

		for img_ext in ['png', 'webp', 'jpg']:
			img_files = list(img_dir.glob(f'*/*_leftImg8bit.{img_ext}'))
			if img_files:
				break

		if not img_files:
			raise FileNotFoundError(f'Did not find images at {img_dir}')


		log.info(f'LAF: found images in {img_ext} format')
		self.img_fmt = img_ext

		# LAF's PNG images contain a gamma value which makes them washed out, ignore it
		# if img_ext == '.png':
			# self.channels['image'].opts['ignoregamma'] = True

		frames = [
			self.laf_id_from_image_path(p)
			for p in img_files
		]
		frames.sort(key = itemgetter('fid'))

		# remove invalid labeled frames
		invalid_indices = self.cfg.get('exclude_frame_indices')
		if invalid_indices is not None:
			valid_indices = np.delete(np.arange(frames.__len__()), invalid_indices)
			frames = [frames[i] for i in valid_indices]

		# remove scenes
		excluded_prefixes = self.cfg.get('exclude_prefix')
		if excluded_prefixes is not None:
			frlen = frames.__len__()
			frames = [
				fr for fr in frames
				if not any([
					fr.fid.startswith(p) for p in excluded_prefixes
				])
			]
			print(f'Exclude {frlen} -> {frames.__len__()}')

		self.set_frames(frames)
		self.check_size()


@DatasetRegistry.register_class()
class DatasetSmallObstacle(DatasetRA):
	# this dataset needs to be preprocessed: create binary lo
	configs = [
		dict(
			name='SmallObstacleDataset-train',
			split='train',
			dir_root=DIR_DATASETS / 'Small_Obstacle_Dataset',
			classes=dict(
				road=0,
				obstacle=1,
				ignore=255,

				usual = 0,
				anomaly = 1,
			),
		),
		dict(
			name='SmallObstacleDataset-test',
			split='test',
			dir_root=DIR_DATASETS / 'Small_Obstacle_Dataset',
			classes=dict(
				road=0,
				obstacle=1,
				ignore=255,

				usual = 0,
				anomaly = 1,
			),
		),
		dict(
			name='SmallObstacleDataset-val',
			split='val',
			dir_root=DIR_DATASETS / 'Small_Obstacle_Dataset',
			classes=dict(
				road=0,
				obstacle=1,
				ignore=255,

				usual = 0,
				anomaly = 1,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/{dset.cfg.split}/{direc}/image/{fid}.png"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/{dset.cfg.split}/{direc}/labels/{fid}.png"),
	}

	def sod_id_from_image_path(cls, path, **_):
		fid = path.stem
		direc = str(fid).split('_')[0] + '_' + str(fid).split('_')[1]
		return EasyDict(
			fid=fid,
			direc=direc
		)

	def discover(self):
		img_dir = Path(self.cfg.dir_root) / self.cfg.split

		for img_ext in ['png', 'webp', 'jpg']:
			img_files = list(img_dir.glob(f'*/labels/*.{img_ext}'))
			if img_files:
				break

		if not img_files:
			raise FileNotFoundError(f'Did not find images at {img_dir}')

		log.info(f'SOD: found images in {img_ext} format')
		self.img_fmt = img_ext

		frames = [
			self.sod_id_from_image_path(p)
			for p in img_files
		]
		frames.sort(key=itemgetter('fid'))

		self.set_frames(frames)
		self.check_size()
