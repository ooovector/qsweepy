import sys, os
import sphinx_rtd_theme

from datetime import datetime
sys.path.insert(0, os.path.abspath('../../../'))
print("this is the path ",os.path.abspath('../../../'))
print('another path: ',os.listdir(os.path.abspath('../../../')))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'nbsphinx',
    'sphinx.ext.mathjax'
              ]
# autosummary_generate = False
# autosummary = []
source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'

project = 'QSweepy'
copyright = ''
author = 'Ilia Besedin'

version = '1'
release = '01012014'

exclude_patterns = ['_build','**.ipynb_checkpoints']
add_function_parentheses = False
autosummary_generate = True

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ['_static']
templates_path = ['_templates']

html_theme_options = {
    'collapse_navigation': True,
    'display_version': True,
}
html_context = {
    'css_files': [
        '_static/theme_overrides.css',  # override wide tables in RTD theme
        ],
     }

def setup(app):
    app.add_css_file("css/style.css")