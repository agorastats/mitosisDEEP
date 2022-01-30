import os

from abc import ABCMeta, abstractmethod
from utils.image import create_dir
from utils.options import ResolveOptions


class CreatePatches(metaclass=ABCMeta):

    def __init__(self, ):
        self._opts = ResolveOptions()
        self.opts = self._opts.get_options()

    def setup(self):
        output_dir = os.path.join(self.opts['output'])
        create_dir(output_dir)
        create_dir(os.path.join(output_dir, self.opts['image_folder']))
        create_dir(os.path.join(output_dir, self.opts['mask_folder']))

    @abstractmethod
    def run(self):
        pass
