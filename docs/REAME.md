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
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

html_theme = "alabaster"

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp'
}
latex_documents = [
    (master_doc, 'EpilDPy.tex', 'The Epidemiology of infectious diseases in Python',
     'Edwin Puertas', 'manual'),
]

man_pages = [
    (master_doc, 'EpilDPy', 'The Epidemiology of infectious diseases in Python',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'EpilDPy', 'The Epidemiology of infectious diseases in Python',
     author, 'Edwin Puertas', 'One line description of project.',
     'Miscellaneous'),
]



step 4
------
cd docs
make html;