import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='my_log_file.txt', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger()

# Create a handler for console output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter for console output (optional)
console_formatter = logging.Formatter('%(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

# Your long-running code goes here
for i in range(10):
    logger.info(f"Processing item {i}")
    print(1)

# You can use different logging levels (e.g., info, warning, error) for different messages

# Don't forget to close the log file handler when done
logging.shutdown()
