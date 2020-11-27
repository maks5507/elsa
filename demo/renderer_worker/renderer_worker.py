#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import jinja2
from .. import noexcept


class RendererWorker:
    def __init__(self, log, main_template_path, models, models_values):
        self.log = log
        self.main_template_path = main_template_path
        self.models = models
        self.models_values = models_values

    @noexcept('')
    def run(self):
        with open(self.main_template_path, 'r') as f:
            main_template = jinja2.Template(f.read())

        data = main_template.render(text='', models=self.models,
                                    model_values=self.models_values)
        return data
