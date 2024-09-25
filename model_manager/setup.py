import subprocess
import sys


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def upgrade(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])

upgrades = ["pip", "wheel", "setuptools", "swig", ]

packages = [
    "ipywidgets", "tensorrt", "tensorflow[and-cuda]",
    "blinker", "mlflow", "optuna", "optuna-integration[mlflow]",
    "plotly", "stable-baselines3", "dagshub", "gymnasium",
    "gymnasium[box2d]", "PyMySQL", "python-dotenv", "stable-baselines3[extra]"
]


for package_upgrade in upgrades:
    print(f"Upgrading {package_upgrade}")
    upgrade(package_upgrade)

for package in packages:
    print(f"Installing {package}")
    install(package)

print("All packages installed successfully!")
