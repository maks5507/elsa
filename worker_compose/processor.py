#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#


class Processor:
    def __init__(self, instance):
        self.instance = instance

    def run(self, chunk, **kwargs):
        for path in chunk:
            self.instance.run(path, **kwargs)
