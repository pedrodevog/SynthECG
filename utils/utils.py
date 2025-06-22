import numpy as np
import torch


def display_config(config, indent=0):
    """
    Recursively display a nested configuration dictionary in a nice, readable format.
    
    Args:
        config (dict): The configuration dictionary to display
        indent (int): Current indentation level (used for recursion)
        indent_char (str): Character(s) to use for indentation
    """
    if indent == 0:
        print("\nConfiguration:")
        print("="*20)

    indent_char='\t'
    for key, value in config.items():
        # Create the current indentation string
        current_indent = indent_char * indent
        
        # If the value is a dictionary, recursively display it
        if isinstance(value, dict):
            print(f"{current_indent}{key}:")
            display_config(value, indent + 1)
        else:
            # Format numerical values nicely
            if isinstance(value, float):
                if value < 0.01:
                    # Use scientific notation for very small values
                    value_str = f"{value:.2e}"
                else:
                    value_str = f"{value:.4f}"
            else:
                value_str = str(value)
                
            print(f"{current_indent}{key}: {value_str}")

    if indent == 0:
        print("="*20)
        print("")
