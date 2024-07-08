# File: logging_utils.py

import logging
import colorlog
from functools import wraps
import time
from typing import Callable, Any, Union
import json

def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration with colored output and improved formatting.
    
    Args:
    level (int): The logging level (e.g., logging.DEBUG, logging.INFO)
    log_file (str, optional): Path to a log file. If None, log to console only.
    """
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(asctime)s%(reset)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
        },
        secondary_log_colors={},
        style='%'
    )

    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = colorlog.getLogger()
    logger.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

def format_dict(d, indent=0):
    """Format a dictionary for pretty printing."""
    return '\n'.join(f"{'  ' * indent}{k}: {format_dict(v, indent+1) if isinstance(v, dict) else v}" for k, v in d.items())

def log_function(logger: logging.Logger):
    """
    A decorator that logs function entry, exit, arguments, and execution time with improved formatting.
    
    Args:
    logger (logging.Logger): The logger to use for logging.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_name = func.__name__
            logger.info(f"{'=' * 40}")
            logger.info(f"Starting: {func_name}")
            
            # Log arguments in a more readable format
            if args or kwargs:
                logger.debug("Arguments:")
                if args:
                    for i, arg in enumerate(args):
                        if isinstance(arg, dict):
                            logger.debug(f"  arg{i}:\n{format_dict(arg, 2)}")
                        else:
                            logger.debug(f"  arg{i}: {arg}")
                if kwargs:
                    for key, value in kwargs.items():
                        if isinstance(value, dict):
                            logger.debug(f"  {key}:\n{format_dict(value, 2)}")
                        else:
                            logger.debug(f"  {key}: {value}")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                logger.info(f"Completed: {func_name}")
                
                # Log the result
                if result:
                    if isinstance(result, dict):
                        logger.info(f"Output:\n{format_dict(result, 1)}")
                    else:
                        logger.info(f"Output: {result}")
                
                return result
            except Exception as e:
                logger.exception(f"Exception in {func_name}:")
                logger.exception(f"  {str(e)}")
                raise
            finally:
                duration = time.time() - start_time
                logger.debug(f"Execution time: {duration:.2f} seconds")
                logger.info(f"{'=' * 40}\n")
        
        return wrapper
    return decorator