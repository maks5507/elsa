#
# Created by Maksim Eremeev (mae9785@nyu.edu)
#

import jinja2
from .. import noexcept


class Renderer:
    def __init__(self, log, main_template_path):
        self.log = log
        self.main_template_path = main_template_path

    @noexcept(default_value={"data": [], "errors": []})
    def run(self, models, models_values):
        with open(self.main_template_path, 'r') as f:
            main_template = jinja2.Template(f.read())

        data = main_template.render(text='', models=models,
                                    model_values=models_values)
        return {'data': [data], 'errors': []}