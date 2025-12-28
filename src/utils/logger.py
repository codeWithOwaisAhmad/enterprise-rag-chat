import logging
import os
from datetime import datetime

def setup_logger(name):
    # 1. Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 2. Prevent duplicate logs
    if logger.hasHandlers():
        return logger
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 3. File Handler (Saves to file)
    log_filename = f"logs/app_{datetime.now().strftime('%Y-%m-%d')}.log"
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(formatter)
    
    # 4. Console Handler (Prints to terminal)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger