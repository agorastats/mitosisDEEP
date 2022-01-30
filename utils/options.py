from optparse import OptionParser


class ResolveOptions(object):

    def __init__(self):
        super().__init__()
        self.opt = OptionParser()
        self.resolve_options()

    def resolve_options(self):
        self.opt.add_option('--img_folder', dest='images_folder',
                            help="folder containing images to patchify", action='store', default='test')
        self.opt.add_option('--mask_folder', dest='masks_folder',
                            help="folder containing images to patchify", action='store', default='test')
        self.opt.add_option('--output', dest='output',
                            help='folder to output images, masks', action='store')
        self.opt.add_option('--data_path', dest='data_path',
                            help='folder containing images')

    def get_options(self):
        (options, args) = self.opt.parse_args()
        return vars(options)


