# Use latest Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run the trading bot
CMD ["python", "main.py"]
