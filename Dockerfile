# Use a smaller base image
FROM python:3.9-slim  

# Set the working directory
WORKDIR /app

# Copy only requirements first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose port 5000 for Flask
EXPOSE 5000

# Start Gunicorn server with 4 worker processes
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
