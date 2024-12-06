# Recommendation System with Streamlit

This project is a basic recommendation system built using Streamlit. It provides an interactive web application that allows users to receive recommendations based on their preferences.

## Project Structure

```
Gui_Recommendation_System
├── src
│   ├── streamlit_rec_sys.py       # Main entry point for the Streamlit application
│   ├── data
│   │   └── load_data.py           # Functions to load and preprocess data
│   ├── models
│   │   └── recommend_model.py      # Recommendation model implementation
│   ├── utils
│   │   └── helpers.py              # Utility functions for the application
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd streamlit-rec-sys
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```
   streamlit run src/streamlit_rec_sys.py
   ```

## Usage Guidelines

- Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
- Follow the on-screen instructions to input your preferences and receive recommendations.

## Overview of the Recommendation System

The recommendation system utilizes various algorithms to analyze user input and provide personalized suggestions. The data loading and preprocessing are handled in `load_data.py`, while the recommendation logic is implemented in `recommend_model.py`. Utility functions that assist with data manipulation are located in `helpers.py`. 

Feel free to explore and modify the code to enhance the recommendation capabilities!