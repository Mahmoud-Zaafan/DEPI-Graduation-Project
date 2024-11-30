# ISEE: A Virtual Assistant for Visually Impaired People

## Overview
ISEE is an innovative virtual assistant designed to enhance the independence and accessibility of visually impaired individuals. The project utilizes technologies such as Neural Networks and Computer Vision to enable users to navigate their surroundings effectively.

## Features
### Implemented Functionalities
- **Distance Estimation**: Calculate distances to objects for better navigation.
- **OCR and Translation**: Read printed text and translate it into the user's preferred language.
- **Object and Color Recognition**: Identify objects and their colors using YOLO and HSV-based methods.
- **Scene Description**: Describe surroundings using a fine-tuned scene interpretation model.
- **NLP Model**: Interpret user commands and queries through a trained Naïve Bayes model with 98% accuracy.

### Excluded Features
- **Currency Detection**: Initially implemented using YOLO but excluded from the final version due to latency and training constraints.

## System Architecture
### Backend Components
1. **Laravel API**: Handles user authentication, request validation, and communication with the AI server.
2. **Flask AI Server**: Processes images and queries, providing intelligent responses.

### Hardware Components
- **3D-Printed Glasses**: Custom-designed for integration with the system.
- **ESP32-CAM Module**: Captures images with compact design and Wi-Fi capabilities.

### Mobile Application
- **Onboarding**: Guided instructions for using the system.
- **Login and Registration**: User verification via name and phone number.
- **OTP Authentication**: Enhances security through one-time password verification.
- **Home Screen**: Primary interface for user interactions.

## Project Directory Structure
```
color recognition/    # Code for object and color recognition using YOLO and HSV
Currency/             # Initial implementation of currency detection
distance/             # Distance estimation code
final presentation/   # Presentation materials for the project
NLP Model/            # NLP model for command interpretation
OCR/                  # OCR implementation for text recognition
Translation-2/        # Translation module for multiple languages
```

## How to Run the Project
1. Clone the repository.
2. Follow setup instructions in the corresponding subdirectory for each feature.
3. Run the Flask server and Laravel API for backend operations.
4. Launch the mobile application for user interaction.

## Future Enhancements
- Real-time processing for all features.
- Support for additional languages in the translation module.
