FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app

# Install the dependencies
RUN pip install -r requirements.txt

#expose the port the app runs on
EXPOSE 8000

# Copy the rest of the application code into the container
COPY . /app

# Command to run the application
CMD ["python3", "app.py"]