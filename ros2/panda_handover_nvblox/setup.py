from glob import glob
from setuptools import find_packages, setup


package_name = "panda_handover_nvblox"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="panda handover experiment",
    maintainer_email="noreply@example.com",
    description="Offline RGB-D adapter for conservative Isaac ROS nvblox ESDF comparison.",
    license="All rights reserved",
    entry_points={
        "console_scripts": [
            "offline_esdf = panda_handover_nvblox.offline_esdf:main",
        ],
    },
)
