from glob import glob
import os

from setuptools import find_packages, setup


package_name = "f1tenth_safety_deployment"
default_model_bundle = "real_track_compact_seed18_qpositive_hybrid300_20260909"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (
            os.path.join("share", package_name, "model_bundles", default_model_bundle),
            glob(os.path.join("model_bundles", default_model_bundle, "*")),
        ),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="F1TENTH safety project contributors",
    maintainer_email="noreply@example.com",
    description="ROS 2 deployment of composed F1TENTH safety policies.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "composed_controller = "
            "f1tenth_safety_deployment.composed_controller_node:main",
            "virtual_opponent = "
            "f1tenth_safety_deployment.virtual_opponent_node:main",
            "export_models = f1tenth_safety_deployment.export_models:main",
        ],
    },
)
