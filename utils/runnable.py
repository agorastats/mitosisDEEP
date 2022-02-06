import logging
import optparse
import sys

from abc import ABCMeta, abstractmethod


class Runnable(object, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def add_options(self, parser):
        pass

    def pre_run(self, options):
        pass

    @abstractmethod
    def run(self, options):
        pass

    def post_run(self, options):
        pass

    def execute(self, options):
        self.pre_run(options)
        self.run(options)
        self.post_run(options)


class Main(object):

    def __init__(self, executable):
        self._executable = executable
        self._options = None
        self.calling_logging()

    @staticmethod
    def calling_logging():
        logging.getLogger().setLevel(logging.INFO)

    def _add_options(self, parser):
        self._executable.add_options(parser)

    def _get_options(self, args):
        parser = optparse.OptionParser()
        self._add_options(parser)
        (options, args) = parser.parse_args(args if args is not None else sys.argv[1:])
        self._options = vars(options)

    def run(self, args=None):
        self._get_options(args)
        try:
            self._executable.execute(self._options)
        except:
            logging.exception('Unexpected error executing', exc_info=True)


class Prova(Runnable):

    def __init__(self):
        super().__init__()
        self.prova = 'prova'

    def add_options(self, parser):
        super().add_options(parser)
        parser.add_option('--test', dest='test', action='store')

    def run(self, options):
        test = options['test']


if __name__ == '__main__':
    Main(Prova()).run()
