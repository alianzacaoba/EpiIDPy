step 1
------
mkdir $APPDIR/docs
cd $APPDIR/docs
sphinx-quickstart

step 2
------
pip install sphinx-rtd-theme

step 3
------
[conf.py]
import os
import sys
sys.path.insert(0, os.path.abspath('../logic'))
extensions = [
'sphinx.ext.todo',
'sphinx.ext.viewcode',
'sphinx.ext.autodoc'
]
import sphinx_glpi_theme

html_theme = "glpi"

html_theme_path = sphinx_glpi_theme.get_html_themes_path()

step 4
------
sphinx-apidoc -f -o docs\source logic


step 5
------
cd docs
make html;