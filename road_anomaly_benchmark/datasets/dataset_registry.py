
import logging
from typing import Callable
from functools import partial

from easydict import EasyDict

log = logging.getLogger(__name__)

class Registry:
	def __init__(self):
		self.INITIALIZERS = {}
		self.MODULES = {}

	def register(self, name : str, init_func : Callable, *opts_s, **opts_kw):
		if opts_s or opts_kw:
			init_func = partial(init_func, *opts_s, **opts_kw)

		if name in self.INITIALIZERS or name in self.MODULES:
			log.warn(f'Module {name} is already registered')
		
		self.INITIALIZERS[name] = init_func

	def register_concrete(self, name : str, dset_object):

		if name in self.MODULES:
			log.warn(f'Module {name} is already registered')
		
		self.MODULES[name] = dset_object

		return dset_object

	def list_available_dsets(self):
		names = set(self.INITIALIZERS.keys()).union(self.MODULES.keys())
		names = list(names)
		names.sort()
		return names

	def get(self, name : str):
		obj = self.MODULES.get(name)
		
		if obj is None:
			init_func = self.INITIALIZERS.get(name)

			if init_func is None:
				dslist = '\n '.join(self.list_available_dsets())
				# KeyError can't display newlines https://stackoverflow.com/questions/46892261/new-line-on-error-message-in-keyerror-python-3-3
				raise ValueError(f'No dataset called {name} in registry, avaiable datasets:\n {dslist}')

			else:
				obj = init_func()
				self.register_concrete(name, obj)

		return obj

	def register_class(self, *args, **kwargs):
		def decorator(class_to_register):
			configs = getattr(class_to_register, 'configs')

			# config generator function
			if isinstance(configs, Callable):
				configs = configs()

			for cfg in configs:
				if isinstance(cfg, dict):
					cfg = EasyDict(cfg)
				self.register(cfg['name'], partial(class_to_register, cfg))	

			return class_to_register

		return decorator


DatasetRegistry = Registry()
