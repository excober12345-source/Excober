import subprocess
import sys

def install(package):
    """Install a package using pip"""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def main():
    # List of packages (matches your requirements.txt)
    packages = [
        "ccxt",
        "python-binance",
        "bybit",
        "okx",
        "alpaca-trade-api",
        "oandapyV20",
        "pandas",
        "numpy",
        "scikit-learn",
        "tensorflow",
        "torch",
        "ta",
        "requests",
        "python-dotenv"
    ]

    for pkg in packages:
        print(f"Installing {pkg}...")
        install(pkg)
    print("✅ All packages installed!")

if __name__ == "__main__":
    main()
