from setuptools import setup

setup(
    name='EpiIDPy',
    version='0.2',
    packages=['logic'],
    url='https://github.com/alianzacaoba/EpiIDPy',
    license='GNU General Public License v3.0',
    author='Edwin Puertas | Angel Paternina-Caicedo',
    author_email='eapuerta@gmail.com',
    description='Epidemiology of infectious diseases in Python (EpilDPy)',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.6.1"
)
