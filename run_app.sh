#!/bin/bash

# Navigate to the directory containing your Flask app
cd ~/Usaid/flask_api/QuestionPro_Test

# Activate the virtual environment
source venv/bin/activate

# Run the Flask app in the background and redirect output to a log file
nohup python3 Contextual_Chat_Bot.py > flask_app.log 2>&1 &

# Get the PID of the background process
FLASK_PID=$!

# Print the process ID of the background process
echo "Flask app is running in the background. Logs are being written to flask_app.log"
echo "Process ID: $FLASK_PID"
