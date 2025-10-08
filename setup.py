# setup with TRACE entrypoint trace_app:main
from setuptools import setup
import pathlib

requirements = pathlib.Path('requirements.txt').read_text().splitlines()

setup(
    name='TRACE',
    version='0.1',
    packages=['trace_app'],
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'trace=trace_app.trace_app:main',
            'trace_desktop_app=trace_app.run_trace_desktop_app:main',
        ],
    },
)
