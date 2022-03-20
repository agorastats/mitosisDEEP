import logging
import optparse
import os
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


class SequenceRunnable(Runnable):
    def __init__(self, *sequence):
        super().__init__()
        self._seq = sequence

    def run(self, options):
        for runnable in self._seq:
            runnable.execute(options)

    def add_options(self, parser):
        for runnable in self._seq:
            runnable.add_options(parser)


class Main(object):

    def __init__(self, executable):
        self._executable = executable
        self._options = None
        self.calling_logging()
        # get relative directory
        os.chdir(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

    @staticmethod
    def calling_logging():
        logging.getLogger().setLevel(logging.INFO)

    def _add_options(self, parser):
        self._executable.add_options(parser)

    def _get_options(self, args):
        parser = optparse.OptionParser(conflict_handler='resolve')
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
        logging.info('__options %s' % str(test))


if __name__ == '__main__':
    Main(Prova()).run()
