

from .dataset_io import DatasetBase, ChannelLoaderImage, imread
from .tracks import DatasetRA
from .dataset_registry import DatasetRegistry
from ..paths import DIR_DATASETS

from pathlib import Path
import json, os

import numpy as np

def read_json(path, key=None, allow_failure=False):
	try:
		file_content = json.loads(Path(path).read_text())
		return file_content[key] if key is not None else file_content

	except Exception as e:
		# print(f'Failed to load JSON {path}: ', e)

		if allow_failure:
			return []
		else:
			raise e


@DatasetRegistry.register_class()
class FishyscapesLAFSubset(DatasetBase):
	DIR_FISHY_LAF = Path(os.environ.get('DIR_FISHY_LAF', DIR_DATASETS / 'dataset_FishyLAF'))

	configs = [
		dict(
			# Fishyscapes subset of LAF dataset, but using original LAF obstacle labels
			name = 'FishyLAFObstacle-val',
			split = 'Obstacle-val',
			expected_length = 98, # 2 frames are removed because their LAF labels are invalid
		),
		dict(
			# Fishyscapes re-labeling
			name = 'FishyLAFAnomaly-val',
			split = 'Anomaly-val',
			expected_length = 100,
			dir_fishy = DIR_FISHY_LAF,
			dir_split = 'val',
		),
		dict(
			# Fishyscapes subset of LAF dataset, but using original LAF obstacle labels
			name = 'FishyLAFObstacle-test',
			split = 'Obstacle-test',
			expected_length = 268, # original 275 minus 7 invalid labels in LAF
		),
	]

	LAF_TEST_FIDS = read_json(
		os.environ.get('FISHY_LAF_ID_FILE', Path('~/FishyLAF-ids.json').expanduser()),
		allow_failure=True,
	)

	FRAME_IDS_SETS = {
		'Anomaly-val': {
			"01_Hanns_Klemm_Str_45_000000_000080",
			"01_Hanns_Klemm_Str_45_000000_000140",
			"01_Hanns_Klemm_Str_45_000000_000200",
			"01_Hanns_Klemm_Str_45_000000_000230",
			"01_Hanns_Klemm_Str_45_000000_000260",
			"01_Hanns_Klemm_Str_45_000001_000060",
			"01_Hanns_Klemm_Str_45_000001_000120",
			"01_Hanns_Klemm_Str_45_000001_000180",
			"01_Hanns_Klemm_Str_45_000001_000210",
			"01_Hanns_Klemm_Str_45_000001_000240", # the LAF labels are invalid
			"01_Hanns_Klemm_Str_45_000002_000070",
			"01_Hanns_Klemm_Str_45_000002_000130",
			"01_Hanns_Klemm_Str_45_000002_000190",
			"01_Hanns_Klemm_Str_45_000002_000220",
			"01_Hanns_Klemm_Str_45_000003_000030",
			"01_Hanns_Klemm_Str_45_000003_000090",
			"01_Hanns_Klemm_Str_45_000003_000150",
			"01_Hanns_Klemm_Str_45_000003_000180",
			"01_Hanns_Klemm_Str_45_000003_000210",
			"01_Hanns_Klemm_Str_45_000004_000120",
			"01_Hanns_Klemm_Str_45_000004_000180",
			"01_Hanns_Klemm_Str_45_000004_000210",
			"01_Hanns_Klemm_Str_45_000005_000080",
			"01_Hanns_Klemm_Str_45_000005_000140",
			"01_Hanns_Klemm_Str_45_000005_000200",
			"01_Hanns_Klemm_Str_45_000005_000230", # the LAF labels are invalid
			"01_Hanns_Klemm_Str_45_000008_000060",
			"01_Hanns_Klemm_Str_45_000008_000120",
			"01_Hanns_Klemm_Str_45_000008_000180",
			"01_Hanns_Klemm_Str_45_000008_000240",
			"01_Hanns_Klemm_Str_45_000009_000090",
			"01_Hanns_Klemm_Str_45_000009_000150",
			"01_Hanns_Klemm_Str_45_000009_000210",
			"01_Hanns_Klemm_Str_45_000010_000050",
			"01_Hanns_Klemm_Str_45_000010_000110",
			"01_Hanns_Klemm_Str_45_000010_000170",
			"01_Hanns_Klemm_Str_45_000010_000200",
			"01_Hanns_Klemm_Str_45_000011_000050",
			"01_Hanns_Klemm_Str_45_000011_000110",
			"01_Hanns_Klemm_Str_45_000011_000170",
			"01_Hanns_Klemm_Str_45_000011_000200",
			"01_Hanns_Klemm_Str_45_000011_000230",
			"01_Hanns_Klemm_Str_45_000011_000260",
			"01_Hanns_Klemm_Str_45_000012_000160",
			"01_Hanns_Klemm_Str_45_000012_000220",
			"01_Hanns_Klemm_Str_45_000012_000250",
			"01_Hanns_Klemm_Str_45_000012_000280",
			"04_Maurener_Weg_8_000000_000030",
			"04_Maurener_Weg_8_000000_000090",
			"04_Maurener_Weg_8_000000_000150",
			"04_Maurener_Weg_8_000000_000180",
			"04_Maurener_Weg_8_000001_000030",
			"04_Maurener_Weg_8_000001_000090",
			"04_Maurener_Weg_8_000001_000150",
			"04_Maurener_Weg_8_000001_000180",
			"04_Maurener_Weg_8_000001_000210",
			"04_Maurener_Weg_8_000002_000060",
			"04_Maurener_Weg_8_000002_000120",
			"04_Maurener_Weg_8_000002_000150",
			"04_Maurener_Weg_8_000003_000030",
			"04_Maurener_Weg_8_000003_000090",
			"04_Maurener_Weg_8_000003_000120",
			"04_Maurener_Weg_8_000004_000020",
			"04_Maurener_Weg_8_000004_000080",
			"04_Maurener_Weg_8_000004_000140",
			"04_Maurener_Weg_8_000004_000170",
			"04_Maurener_Weg_8_000004_000200",
			"04_Maurener_Weg_8_000005_000050",
			"04_Maurener_Weg_8_000005_000110",
			"04_Maurener_Weg_8_000005_000170",
			"04_Maurener_Weg_8_000005_000200",
			"04_Maurener_Weg_8_000006_000040",
			"04_Maurener_Weg_8_000006_000100",
			"04_Maurener_Weg_8_000006_000130",
			"04_Maurener_Weg_8_000007_000020",
			"04_Maurener_Weg_8_000007_000080",
			"04_Maurener_Weg_8_000007_000140",
			"04_Maurener_Weg_8_000007_000170",
			"04_Maurener_Weg_8_000008_000040",
			"04_Maurener_Weg_8_000008_000100",
			"04_Maurener_Weg_8_000008_000130",
			"04_Maurener_Weg_8_000008_000160",
			"04_Maurener_Weg_8_000008_000190",
			"13_Elly_Beinhorn_Str_000000_000050",
			"13_Elly_Beinhorn_Str_000000_000110",
			"13_Elly_Beinhorn_Str_000000_000170",
			"13_Elly_Beinhorn_Str_000000_000230",
			"13_Elly_Beinhorn_Str_000000_000260",
			"13_Elly_Beinhorn_Str_000000_000290",
			"13_Elly_Beinhorn_Str_000001_000080",
			"13_Elly_Beinhorn_Str_000001_000140",
			"13_Elly_Beinhorn_Str_000001_000200",
			"13_Elly_Beinhorn_Str_000001_000230",
			"13_Elly_Beinhorn_Str_000002_000030",
			"13_Elly_Beinhorn_Str_000002_000090",
			"13_Elly_Beinhorn_Str_000002_000120",
			"13_Elly_Beinhorn_Str_000003_000080",
			"13_Elly_Beinhorn_Str_000003_000140",
			"13_Elly_Beinhorn_Str_000003_000200",
			"13_Elly_Beinhorn_Str_000003_000260",
		},
		'Anomaly-test': set(LAF_TEST_FIDS['test'] if LAF_TEST_FIDS else []),
	}

	# remove frames with invalid LAF labels
	FRAME_IDS_SETS['Obstacle-val'] = FRAME_IDS_SETS['Anomaly-val'].difference({
		"01_Hanns_Klemm_Str_45_000001_000240",
		"01_Hanns_Klemm_Str_45_000005_000230",
	})

	if FRAME_IDS_SETS['Anomaly-test']:
		FRAME_IDS_SETS['Obstacle-test'] = FRAME_IDS_SETS['Anomaly-test'].difference(
			LAF_TEST_FIDS['testInvalid']
		)

	FRAME_IDS = {
		split: list(sorted(fidset))
		for split, fidset in FRAME_IDS_SETS.items()
	}

	channels_extra = {
		'semantic_class_gt': ChannelLoaderImage(
			'{dset.cfg.dir_fishy}/labels/{dset.cfg.dir_split}/{fid}_fishy-labels.png',
		),
	}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()

	@property
	def fids(self):
		return self.FRAME_IDS[self.cfg.split]

	@property
	def b_load_labels(self):
		return self.cfg.get('dir_fishy') is not None

	def discover(self):
		self.laf_dsets = {
			'train': DatasetRegistry.get('LostAndFound-train'),
			'test': DatasetRegistry.get('LostAndFound-test'),
		}

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, str): # by fid
			if idx_or_fid not in self.fids:
				raise KeyError(f'Id {idx_or_fid} is not in this dataset {self}')
			else:
				fid = idx_or_fid
		else:
			fid = self.FRAME_IDS[self.cfg.split][idx_or_fid]

		if fid in self.laf_dsets['train'].frames_by_fid:
			base_ds = self.laf_dsets['train']
		else:
			base_ds = self.laf_dsets['test']

		key_labels = 'semantic_class_gt'
		if self.b_load_labels:
			channels = set(channels).difference({key_labels, 'semantic_class_gt'})

		fr = base_ds.get_frame(fid, *channels)

		if self.b_load_labels:
			label = self.channels_extra[key_labels].read(dset=self, **fr)
			fr[key_labels] = label
			fr['label_pixel_gt'] = label

		return fr

	def __len__(self):
		return self.fids.__len__()


@DatasetRegistry.register_class()
class LostAndFoundAnomaly(DatasetBase):
	"""
	LAF obstacles, but the ROI is the whole image
	(except ego vehicle and sensor artifacts)
	not just the drivable space.
	"""
	configs = [
		dict(
			name = 'LostAndFound-anomalyTrain',
			base_ds = 'LostAndFound-train',
			roi_img = Path(__file__).with_name('LAF_roi_2048.png'),
		),
		dict(
			name = 'LostAndFound-anomalyTest',
			base_ds = 'LostAndFound-test',
			roi_img = Path(__file__).with_name('LAF_roi_2048.png'),
		),
	]

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()

	def discover(self):
		self.base_ds = DatasetRegistry.get(self.cfg.base_ds)
		self.roi_mask = imread(self.cfg.roi_img).astype(bool)

	def get_frame(self, idx_or_fid, *channels):
		fr = self.base_ds.get_frame(idx_or_fid, *channels)

		pix_gt = fr.get('label_pixel_gt')
		if pix_gt is not None:
			h, w = pix_gt.shape[:2]
			label = np.full((h, w), 255, dtype=np.uint8)
			label[self.roi_mask] = 0
			label[pix_gt == 1] = 1
			fr['label_pixel_gt'] = label

		return fr

	def __len__(self):
		return self.base_ds.__len__()



@DatasetRegistry.register_class()
class ErasingSubset(DatasetBase):
	configs = [
		dict(
			# from article "Detecting Road Obstacles By Erasing Them" 2020 version
			name = 'Erasing-20',
			split = 'Erasing20',
			expected_length = 105,
			name_for_persistence = 'ObstacleTrack-all',
		),
		dict(
			# from article "Detecting Road Obstacles By Erasing Them" 2020 version, without the dog
			name = 'Erasing-20nodog',
			split = 'Erasing20-no-dog',
			expected_length = 100,
			name_for_persistence = 'ObstacleTrack-all',
		),

		dict(
			# from article "Detecting Road Obstacles By Erasing Them", new snowstorm sequence
			name = 'Erasing-21',
			split = 'Erasing21',
			expected_length = 155,
			name_for_persistence = 'ObstacleTrack-all',
		),
	]

	FRAME_IDS = {
		'Erasing20-no-dog': [
			"darkasphalt_axestump_2",
			"darkasphalt_axestump_3",
			"darkasphalt_bag_3",
			"darkasphalt_basket_2",
			"darkasphalt_basket_3",
			"darkasphalt_boot_2",
			"darkasphalt_boot_3",
			"darkasphalt_bottles_1",
			"darkasphalt_bottles_2",
			"darkasphalt_bucket_3",
			"darkasphalt_bucket_4",
			"darkasphalt_canister_2",
			"darkasphalt_canister_3",
			"darkasphalt_cans_1",
			"darkasphalt_cans_2",
			"darkasphalt_helmetO_1",
			"darkasphalt_helmetO_2",
			"darkasphalt_stump_1",
			"darkasphalt_stump_2",
			"darkasphalt_watercanS_2",
			"gravel_axestump_2",
			"gravel_axestump_3",
			"gravel_bag_3",
			"gravel_basket_2",
			"gravel_basket_3",
			"gravel_boot_1",
			"gravel_boot_2",
			"gravel_bottle_2",
			"gravel_bottle_3",
			"gravel_bottle_4",
			"gravel_bucket_3",
			"gravel_canister_3",
			"gravel_canister_4",
			"gravel_cans2_2",
			"gravel_helmetG_2",
			"gravel_log_2",
			"gravel_log_3",
			"gravel_stump_1",
			"gravel_stump_2",
			"gravel_watercanS_2",
			"greyasphalt_axestump_2",
			"greyasphalt_axestump_3",
			"greyasphalt_bag_3",
			"greyasphalt_bag_4",
			"greyasphalt_basket_3",
			"greyasphalt_basket_4",
			"greyasphalt_boot_2",
			"greyasphalt_boot_3",
			"greyasphalt_bottle_1",
			"greyasphalt_bottle_2",
			"greyasphalt_canister_2",
			"greyasphalt_canister_3",
			"greyasphalt_cansA_1",
			"greyasphalt_cansB_2",
			"greyasphalt_helmetB_1",
			"greyasphalt_helmetB_2",
			"greyasphalt_stump_2",
			"greyasphalt_stump_3",
			"greyasphalt_watercanB_4",
			"greyasphalt_watercanS_2",
			"motorway_axestump_2",
			"motorway_basket_1",
			"motorway_basket_4",
			"motorway_boot_1",
			"motorway_boot_3",
			"motorway_boot_6_and_bird",
			"motorway_bottles_1",
			"motorway_bottles_k",
			"motorway_bucket_3",
			"motorway_canister_3",
			"motorway_cans_1",
			"motorway_helmetB_1",
			"motorway_helmetB_3",
			"motorway_helmetO_1",
			"motorway_stone_1",
			"motorway_stone_2",
			"motorway_stump_2",
			"motorway_stump_3",
			"motorway_triangle_2",
			"motorway_watercanS_1",
			"paving_axe_2",
			"paving_axestump_2",
			"paving_bag_3",
			"paving_bag_6",
			"paving_basket_3",
			"paving_basket_5",
			"paving_boot_1",
			"paving_boot_2",
			"paving_bottles_1",
			"paving_bottles_2",
			"paving_bucket_3",
			"paving_bucket_6",
			"paving_canister_4",
			"paving_cans_2",
			"paving_helmetB_1",
			"paving_helmetB_2",
			"paving_stump_2",
			"paving_watercanS_2",
			"paving_wood_2",
			"paving_wood_3",
		],
		'Erasing20-dog': [
			"darkasphalt2_dog_1",
			"darkasphalt2_dog_2",
			"darkasphalt2_dog_3",
			"darkasphalt2_dog_4",
			"darkasphalt2_dog_5",
		],
		'Erasing21-snowstorm': [
			'snowstorm1_00_16_02.578',
			'snowstorm1_00_14_05.595',
			'snowstorm1_00_14_37.727',
			'snowstorm1_00_03_05.185',
			'snowstorm2_00_02_32.753',
			'snowstorm2_00_06_00.260',
			'snowstorm1_00_11_28.972',
			'snowstorm1_00_15_56.322',
			'snowstorm1_00_00_56.106',
			'snowstorm1_00_10_55.939',
			'snowstorm2_00_04_07.881',
			'snowstorm1_00_00_51.868',
			'snowstorm2_00_02_30.667',
			'snowstorm1_00_15_20.970',
			'snowstorm2_00_01_39.950',
			'snowstorm1_00_13_15.261',
			'snowstorm1_00_07_41.845',
			'snowstorm1_00_03_24.738',
			'snowstorm2_00_04_46.303',
			'snowstorm2_00_05_55.972',
			'snowstorm1_00_08_26.106',
			'snowstorm1_00_14_02.892',
			'snowstorm2_00_00_49.600',
			'snowstorm1_00_16_00.827',
			'snowstorm2_00_05_47.380',
			'snowstorm1_00_13_18.748',
			'snowstorm2_00_06_46.423',
			'snowstorm1_00_07_43.930',
			'snowstorm2_00_06_49.910',
			'snowstorm1_00_10_52.886',
			'snowstorm1_00_15_24.040',
			'snowstorm1_00_14_40.363',
			'snowstorm1_00_10_48.898',
			'snowstorm1_00_03_08.939',
			'snowstorm1_00_08_21.968',
			'snowstorm2_00_01_35.045',
			'snowstorm2_00_03_09.556',
			'snowstorm2_00_02_27.047',
			'snowstorm2_00_04_52.192',
			'snowstorm1_00_08_18.898',
			'snowstorm2_00_04_14.087',
			'snowstorm2_00_07_27.397',
			'snowstorm1_00_10_06.489',
			'snowstorm2_00_03_13.443',
			'snowstorm1_00_11_33.009',
			'snowstorm2_00_04_12.152',
			'snowstorm2_00_00_52.502',
			'snowstorm2_00_06_54.030',
			'snowstorm2_00_04_49.172',
			'snowstorm2_00_01_38.181',
			'snowstorm1_00_09_22.178',
			'snowstorm1_00_15_26.042',
			'snowstorm2_00_06_52.545',
			'snowstorm2_00_05_53.503',
			'snowstorm1_00_00_59.226',
		],
	}

	FRAME_IDS['Erasing20'] = FRAME_IDS['Erasing20-no-dog'] + FRAME_IDS['Erasing20-dog']
	FRAME_IDS['Erasing21'] = FRAME_IDS['Erasing20'] + FRAME_IDS['Erasing21-snowstorm']

	for fidlist in FRAME_IDS.values():
		fidlist.sort()

	FRAME_IDS_SETS = {
		# split: {
		# 	fid: i for i, fid in enumerate(fids)
		# }
		split: set(fids)
		for split, fids in FRAME_IDS.items()
	}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()

	@property
	def fids(self):
		return self.FRAME_IDS[self.cfg.split]

	def discover(self):
		self.base_ds = DatasetRegistry.get('ObstacleTrack-all')

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, str): # by fid
			if idx_or_fid not in self.fids:
				raise KeyError(f'Id {idx_or_fid} is not in this dataset {self}')
			else:
				fid = idx_or_fid
		else:
			fid = self.FRAME_IDS[self.cfg.split][idx_or_fid]

		fr = self.base_ds.get_frame(fid, *channels)
		return fr

	def __len__(self):
		return self.fids.__len__()

@DatasetRegistry.register_class()
class RoadAnomalyByClass(DatasetRA):
	configs = [
		dict(
			name = 'AnomalyTrack-animals',
			img_fmt = 'jpg',
			split = 'animals',
			dir_root = DIR_DATASETS / 'dataset_AnomalyTrack',
			expected_length = 59,
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 255,
			),
		),
		dict(
			name = 'AnomalyTrack-vehicles',
			img_fmt = 'jpg',
			split = 'vehicles',
			dir_root = DIR_DATASETS / 'dataset_AnomalyTrack',
			expected_length = 23,
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 255,
			),
		),
		dict(
			name = 'AnomalyTrack-other',
			img_fmt = 'jpg',
			split = 'other',
			dir_root = DIR_DATASETS / 'dataset_AnomalyTrack',
			expected_length = 11,
			classes = dict(
				usual = 0,
				anomaly = 1,
				ignore = 255,
			),
		),
		dict(
			name='AnomalyTrack-validation',
			img_fmt='jpg',
			split='validation',
			dir_root=DIR_DATASETS / 'dataset_AnomalyTrack',
			expected_length=10,
			classes=dict(
				usual=0,
				anomaly=1,
				ignore=255,
			),
		),
		dict(
			name='AnomalyTrack-test',
			img_fmt='jpg',
			split='test',
			dir_root=DIR_DATASETS / 'dataset_AnomalyTrack',
			expected_length=100,
			classes=dict(
				usual=0,
				anomaly=1,
				ignore=255,
			),
		),
	]

	channels = {
		'image': ChannelLoaderImage("{dset.cfg.dir_root}/images/{fid}.{dset.cfg.img_fmt}"),
		'semantic_class_gt': ChannelLoaderImage("{dset.cfg.dir_root}/labels_masks/{fid}_labels_semantic.png"),
	}

	FRAME_IDS = {
		'animals': [
			'zebra0000',
			'zebra0001',
			'sheep0000',
			'sheep0001',
			'sheep0002',
			'sheep0003',
			'sheep0004',
			'sheep0005',
			'sheep0006',
			'sheep0007',
			'sheep0008',
			'sheep0009',
			'sheep0010',
			'rhino0000',
			'rhino0001',
			'pig0000',
			'pig0001',
			'lion0000',
			'leopard0000',
			'horse0000',
			'horse0001',
			'horse0002',
			'horse0003',
			'horse0004',
			'goose0000',
			'elephant0000',
			'elephant0001',
			'elephant0002',
			'elephant0003',
			'elephant0004',
			'elephant0005',
			'elephant0006',
			'elephant0007',
			'elephant0008',
			'elephant0009',
			'elephant0010',
			'elephant0011',
			'donkey0000',
			'donkey0001',
			'deer0000',
			'deer0001',
			'cow0000',
			'cow0001',
			'cow0002',
			'cow0003',
			'cow0004',
			'cow0005',
			'cow0006',
			'cow0007',
			'cow0008',
			'cow0009',
			'cow0010',
			'cow0011',
			'cow0012',
			'cow0013',
			'cock0000',
			'camel0000',
			'camel0001',
			'bear0000',
		],
		'vehicles': [
			'airplane0000',
			'airplane0001',
			'boat_trailer0000',
			'boat_trailer0001',
			'boat_trailer0002',
			'boat_trailer0003',
			'boat_trailer0004',
			'caravan0000',
			'caravan0001',
			'caravan0002',
			'caravan0003',
			'caravan0004',
			'caravan0005',
			'caravan0006',
			'caravan0007',
			'carriage0001',  # only carriage where horse is not visible
			'scooter0000',
			'tractor0000',
			'tractor0001',
			'tractor0002',
			'tractor0003',
			'tractor0004',
			'tractor0005',
		],
		'other': [
			'coffin0000',
			'cones0000',
			'cones0001',
			'cones0002',
			'cones0003',
			'cones0004',
			'cones0005',
			'cones0006',
			'hay0000',
			'piano0000',
			'tent0000',
		],
		'validation': [
			'validation0000',
			'validation0001',
			'validation0002',
			'validation0003',
			'validation0004',
			'validation0005',
			'validation0006',
			'validation0007',
			'validation0008',
			'validation0009',
		],
		'test': [
			'zebra0000',
			'zebra0001',
			'sheep0000',
			'sheep0001',
			'sheep0002',
			'sheep0003',
			'sheep0004',
			'sheep0005',
			'sheep0006',
			'sheep0007',
			'sheep0008',
			'sheep0009',
			'sheep0010',
			'rhino0000',
			'rhino0001',
			'pig0000',
			'pig0001',
			'lion0000',
			'leopard0000',
			'horse0000',
			'horse0001',
			'horse0002',
			'horse0003',
			'horse0004',
			'goose0000',
			'elephant0000',
			'elephant0001',
			'elephant0002',
			'elephant0003',
			'elephant0004',
			'elephant0005',
			'elephant0006',
			'elephant0007',
			'elephant0008',
			'elephant0009',
			'elephant0010',
			'elephant0011',
			'donkey0000',
			'donkey0001',
			'deer0000',
			'deer0001',
			'cow0000',
			'cow0001',
			'cow0002',
			'cow0003',
			'cow0004',
			'cow0005',
			'cow0006',
			'cow0007',
			'cow0008',
			'cow0009',
			'cow0010',
			'cow0011',
			'cow0012',
			'cow0013',
			'cock0000',
			'camel0000',
			'camel0001',
			'bear0000',
			'airplane0000',
			'airplane0001',
			'boat_trailer0000',
			'boat_trailer0001',
			'boat_trailer0002',
			'boat_trailer0003',
			'boat_trailer0004',
			'caravan0000',
			'caravan0001',
			'caravan0002',
			'caravan0003',
			'caravan0004',
			'caravan0005',
			'caravan0006',
			'caravan0007',
			'carriage0000',
			'carriage0001',  # only carriage where horse is not visible
			'carriage0002',
			'carriage0003',
			'carriage0004',
			'carriage0005',
			'carriage0006',
			'carriage0007',
			'scooter0000',
			'tractor0000',
			'tractor0001',
			'tractor0002',
			'tractor0003',
			'tractor0004',
			'tractor0005',
			'coffin0000',
			'cones0000',
			'cones0001',
			'cones0002',
			'cones0003',
			'cones0004',
			'cones0005',
			'cones0006',
			'hay0000',
			'piano0000',
			'tent0000',
		],
	}

	for fidlist in FRAME_IDS.values():
		fidlist.sort()

	FRAME_IDS_SETS = {
		split: set(fids)
		for split, fids in FRAME_IDS.items()
	}

	def __init__(self, cfg):
		super().__init__(cfg)
		self.discover()
		super().check_size()

	@property
	def fids(self):
		return self.FRAME_IDS[self.cfg.split]

	def discover(self):
		self.base_ds = DatasetRegistry.get('AnomalyTrack-all')

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, str): # by fid
			if idx_or_fid not in self.fids:
				raise KeyError('Id {idx_or_fid} is not in this dataset {self}')
			else:
				fid = idx_or_fid
		else:
			fid = self.FRAME_IDS[self.cfg.split][idx_or_fid]

		fr = self.base_ds.get_frame(fid, *channels)
		return fr

	def __len__(self):
		return self.fids.__len__()
