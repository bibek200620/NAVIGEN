from setuptools import find_packages, setup

package_name = 'navigen_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/perception.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NAVIGEN Team',
    maintainer_email='team@navigen.local',
    description='Traversability segmentation with swappable backends.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'traversability_node = navigen_perception.traversability_node:main',
        ],
    },
)
