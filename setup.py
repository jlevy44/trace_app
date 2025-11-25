# setup with TRACE entrypoint trace_app:main
from setuptools import setup
import pathlib

requirements = pathlib.Path('requirements.txt').read_text().splitlines()

long_description = pathlib.Path('README.md').read_text()

setup(
    name='TRACE',
    version='0.2',
    packages=['trace_app'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'trace=trace_app.trace_app:main',
            'trace_desktop_app=trace_app.run_trace_desktop_app:main',
            'trace_postprocessing=trace_app.postprocessing:main',
        ],
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
)
