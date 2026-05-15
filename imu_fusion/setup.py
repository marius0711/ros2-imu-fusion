from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'imu_fusion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='marius',
    maintainer_email='marius.zepf17@web.de',
    description='EKF-based IMU orientation estimation as a ROS2 node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_fusion_node = imu_fusion.imu_fusion_node:main',
        ],
    },
)
