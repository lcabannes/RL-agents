import re
import math

def calculator(expression):
    
    allowed_chars = r'[\d+\-*/()\.\sπ]|sqrt|sin|cos|tan|log'
    if not re.match(f'^({allowed_chars})+$', expression):
        raise ValueError("Error : Invalide expression")
    
    try:
        result = eval(expression, {"__builtins__": None}, {"log": math.log, "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "π": math.pi})
        return result
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")