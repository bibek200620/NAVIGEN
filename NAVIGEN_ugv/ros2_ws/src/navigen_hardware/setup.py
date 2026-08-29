from setuptools import find_packages, setup

package_name = 'navigen_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/hardware.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NAVIGEN Team',
    maintainer_email='team@navigen.local',
    description='Skid-steer kinematics, ESP32 serial protocol and bridge node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'esp32_bridge = navigen_hardware.esp32_bridge_node:main',
        ],
    },
)
