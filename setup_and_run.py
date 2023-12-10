import subprocess
import sys

def install_packages():
    """
    Install required Python packages from requirements.txt.
    """
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def download_nltk_resources():
    """
    Download necessary NLTK resources.
    """
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

def run_application():
    """
    Run the main application script.
    """
    import main
    main.main()

if __name__ == "__main__":
    print("Installing required packages...")
    install_packages()

    print("Downloading NLTK resources...")
    download_nltk_resources()

    print("Running the application...")
    run_application()