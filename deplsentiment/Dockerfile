# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install flask nltk matplotlib wordcloud

# Download NLTK data (VADER lexicon)
RUN python -m nltk.downloader vader_lexicon

# Expose the port Flask runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
