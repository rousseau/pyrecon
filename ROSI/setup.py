from setuptools import setup
setup(
    name='ROSI',
    version='0.0.1',
    entry_points={
        'console_scripts' :[
            'run_registration=rosi.run_registration_function:main',
        ]
    }
)