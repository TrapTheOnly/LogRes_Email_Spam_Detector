![image with computer and emails](https://github.com/TrapTheOnly/LogRes_Email_Spam_Detector/blob/main/cover.png "Cover Image")

# Spam Email Detection System

## Project Overview
This Spam Email Detection System is a machine learning-based application designed to classify emails into 'Spam' and 'Not Spam' categories. It utilizes a Logistic Regression model trained on a comprehensive dataset obtained from [Kaggle](https://www.kaggle.com/datasets/venky73/spam-mails-dataset), and incorporates advanced text processing techniques to achieve high accuracy and efficiency. The system is particularly adept at handling various email formats, distinguishing between responses and new messages, and adapting to the nuances of email communication.

## Key Features
- **Advanced Text Processing**: Implements cleaning, tokenization, and stemming for optimal text analysis.
- **Feature Engineering**: Includes custom features like response detection to enhance classification accuracy.
- **Logistic Regression Model**: Utilizes Logistic Regression for reliable and interpretable spam detection.
- **Interactive Terminal Interface**: Offers a user-friendly terminal interface for easy input and prediction of email data.
- **Cross-Validation**: Employs cross-validation techniques for robust model evaluation.

## Technologies Used
- **Python**: The core programming language used for the implementation.
- **Pandas & NumPy**: For efficient data manipulation and numerical operations.
- **SciPy**: To handle sparse matrix operations, enhancing the efficiency of feature handling.
- **Scikit-learn**: For machine learning model development, training, and evaluation.
- **NLTK**: Used for natural language processing tasks like stopwords removal and tokenization.
- **TfidfVectorizer**: For converting the email text into a meaningful vector of numbers.

## How to Run the Project
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/TrapTheOnly/LogRes_Email_Spam_Detector.git
   ```
2. **Navigate to the Project Directory**:
   ```bash
   cd LogRes_Email_Spam_Detector
   ```
3. **Run the Setup Script**:
   ```bash
   python setup_and_run.py
   ```
   This script will install necessary dependencies, download required NLTK resources, and launch the application.

## Usage
Once the application is running, follow the on-screen prompts to enter the subject and body of the email. After entering the email details, the system will analyze and display whether the email is classified as Spam or Not Spam.

## Contributions
Contributions, issues, and feature requests are welcome. Feel free to check [issues page](https://github.com/TrapTheOnly/LogRes_Email_Spam_Detector/issues) if you want to contribute.

## License
Distributed under the MIT License. See `LICENSE` for more information.
