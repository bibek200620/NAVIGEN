from setuptools import find_packages, setup

package_name = 'navigen_safety'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/safety.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NAVIGEN Team',
    maintainer_email='team@navigen.local',
    description='Safety supervisor for the NAVIGEN UGV.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'safety_supervisor = navigen_safety.safety_supervisor_node:main',
        ],
    },
)
