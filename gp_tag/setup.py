from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'gp_tag'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Config files
        (os.path.join('share', package_name, 'config', 'camera'), glob('config/camera/*.yaml')),
        # Model/Data files
        (os.path.join('share', package_name, 'data'), ['gp_tag/tag3_blank_360.png']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='S. E. Sundén Byléhn',
    maintainer_email='s.e.sunden.bylehn@gmail.com',
    description='Global Positioning Tag (GP-Tag) detection and localization package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gp_tag_camera = gp_tag.gp_tag_camera:main',
            'gp_tag_realsense = gp_tag.gp_tag_realsense:main',
            'gp_tag_node = gp_tag.gp_tag_node:main',
            'gp_tag_viewer = gp_tag.gp_tag_viewer:main',
            # SLAM integration planned for future release
            # 'gp_tag_slam = gp_tag.gp_tag_slam_node:main',
        ],
    },
    python_requires='>=3.8'
)