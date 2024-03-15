from setuptools import setup

setup(
    name='pygsm',
    version='0.1.0',
    description='A gaussian version of PySM',
    license='MIT',
    packages=['pygsm', 'pygsm.data.cmb_spec'],
    include_package_data=True,
    install_requires=['numpy', 'healpy'],
    author='Yiqi Liu',
    author_email='andrew.liu@princeton.edu',
    url='http://github.com/liuyiqiandrew/pygsm'
)