 
from pathlib import Path
from operator import itemgetter
import logging

from easydict import EasyDict
import numpy as np
import h5py
from ..jupyter_show_image import imread, imwrite

log = logging.getLogger(__name__)

class ChannelLoader:
	def read(self, **frame):
		raise NotImplementedError()

	def write(self, value, **frame):
		raise NotImplementedError()

class ChannelLoaderFileCollection:
	"""
	The channel's value for each frame is in a separate file, for example an image.
	@param file_path_tmpl: a template string or a function(channel, **frame)
	"""
	def __init__(self, file_path_tmpl):
		# convert Path to str so that we can .format it
		self.file_path_tmpl = str(file_path_tmpl) if isinstance(file_path_tmpl, Path) else file_path_tmpl

	def resolve_template(self, **frame):
		template = self.file_path_tmpl
		if isinstance(template, str):
			# string template
			return template.format(
				**frame,
			)
		else:
			# function template
			return template(channel=self, **frame)

	def read(self, **frame):
		path = Path(self.resolve_template(**frame))
		return self.read_file(path)

	def write(self, value, **frame):
		path = Path(self.resolve_template(**frame))
		path.parent.mkdir(exist_ok=True, parents=True)
		self.write_file(path, value)

	def read_file(self, path):
		raise NotImplementedError('read_file for {c}'.format(c=self.__class__.__name__))

	def write_file(self, path, data):
		raise NotImplementedError('write_file for {c}'.format(c=self.__class__.__name__))

	def __repr__(self):
		return '{cls}({tp})'.format(cls=self.__class__.__name__, tp=self.file_path_tmpl)


# class ImageBackgroundService:
# 	IMWRITE_BACKGROUND_THREAD = ThreadPoolExecutor(max_workers=3)

# 	@classmethod
# 	def imwrite(cls, path, data):
# 		cls.IMWRITE_BACKGROUND_THREAD.submit(imwrite, path, data)


class ChannelLoaderImage(ChannelLoaderFileCollection):
	def __init__(self, file_path_tmpl=None, save_opts={}, drop_alpha=True):
		super().__init__(file_path_tmpl)
		self.save_opts = save_opts
		self.drop_alpha = drop_alpha

	def read_file(self, path):
		img_data = imread(path, **self.save_opts)

		if img_data.shape.__len__() > 2 and img_data.shape[2] == 4 and self.drop_alpha:
			img_data = img_data[:, :, :3]

		return img_data

	def write_file(self, path, data):
		imwrite(path, data)


class ChannelLoaderHDF5(ChannelLoaderFileCollection):
	def __init__(self, file_path_tmpl=None, var_name='value', compression=None):
		super().__init__(file_path_tmpl)
		self.var_name = var_name
		self.compression = compression

	@staticmethod
	def read_hdf5_variable(variable):
		if variable.shape.__len__() > 0:
			return variable[:]
		else:
			return variable

	def read_file(self, path):
		var_name = self.var_name

		with h5py.File(path, 'r') as hdf5_file_handle:
			try:
				return self.read_hdf5_variable(hdf5_file_handle[var_name])
			except KeyError as e:
				raise KeyError(f'Failed to read {var_name} from handle with keys {hdf5_file_handle.keys()}: {e}')
		
	def write_file(self, path, data):
		var_name = self.var_name

		path = Path(path)
		path.parent.mkdir(exist_ok=True, parents=True)

		with h5py.File(path, 'w') as hdf5_file_handle:
			if var_name in hdf5_file_handle:
				hdf5_file_handle[var_name][:] = data
			else:
				hdf5_file_handle.create_dataset(var_name, data=data, compression=self.compression)


def hdf5_write_hierarchy_to_group(group, hierarchy):
	for name, value in hierarchy.items():
		# sub-dict
		if isinstance(value, dict):
			hdf5_write_hierarchy_to_group(
				group = group.create_group(name), 
				hierarchy = value
			)
		# label or single value
		elif isinstance(value, (str, bytes, float, int)):
			group.attrs[name] = value
		# ndarray
		elif isinstance(value, np.ndarray):
			group[name] = value
		else:
			raise TypeError(f'Failed to write type {type(value)} to hdf: {name}={value}')
			
def hdf5_write_hierarchy_to_file(path, hierarchy, create_parent_dir=True):
	if create_parent_dir:
		path = Path(path)
		path.parent.mkdir(exist_ok=True, parents=True)

	with h5py.File(path, 'w') as f:
		hdf5_write_hierarchy_to_group(f, hierarchy)
	
def hdf5_read_hierarchy_from_group(group):
	return EasyDict(
		# label or single value
		**group.attrs,
		# numeric arrays
		**{
			name: hdf5_read_hierarchy_from_group(value) 
			if isinstance(value, h5py.Group) else value[()]
			for name, value in group.items()
		}
	)
	
def hdf5_read_hierarchy_from_file(path):
	with h5py.File(path, 'r') as f:
		return hdf5_read_hierarchy_from_group(f)
		



class DatasetBase:

	def __init__(self, cfg):
		self.cfg = EasyDict(cfg)
		self.frames = []
		self.frames_by_fid = {}

	@property
	def name(self):
		return self.cfg.name

	def __str__(self):
		return self.cfg.name

	def set_frames(self, frame_list):
		self.frames = list(frame_list)
		self.frames.sort(key = itemgetter('fid'))
		self.frames_by_fid = {fr['fid']: fr for fr in self.frames}

	def get_frame(self, idx_or_fid, *channels):
		if isinstance(idx_or_fid, int):
			fr = self.frames[idx_or_fid]
		else:
			fr = self.frames_by_fid[idx_or_fid]
		
		out_fr = EasyDict(fr, dset_name = self.cfg.get('name_for_persistence', self.cfg.name))

		channels = channels or self.channels.keys()

		for ch_name in channels:
			try:
				out_fr[ch_name] = self.channels[ch_name].read(dset=self, **fr)
			except FileNotFoundError:
				out_fr[ch_name] = None

		return out_fr
	
	def check_size(self):
		desired_len = self.cfg.get('expected_length')
		actual_len = self.__len__()

		if desired_len is not None and actual_len != desired_len:
			raise ValueError(f'{self}: The dataset should have {desired_len} frames but found {actual_len}')

	def iter(self, *channels):
		"""
		Iterate over frames but only load a subset of channels. For example:
			for fr in dset.iter('image'):
				process(fr.image)
		"""

		class It:
			def __init__(self, dset, channels):
				self.dset = dset
				self.channels = channels

			def __len__(self):
				return self.dset.__len__()

			def __iter__(self):
				for idx in range(self.__len__()):
					yield self.dset.get_frame(idx, *self.channels)
		
		return It(self, channels)	


	def __getitem__(self, idx_or_fid):
		return self.get_frame(idx_or_fid)

	def __len__(self):
		return self.frames.__len__()

	def __iter__(self):
		for idx in range(self.__len__()):
			yield self[idx]
