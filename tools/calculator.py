import re

def calculator(expression):
    
    allowed_chars = r'[\d+\-*/()\.\sπ]|sqrt|sin|cos|tan|log'
    if not re.match(f'^({allowed_chars})+$', expression):
        raise ValueError("Error : Invalide expression")
    
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return result
    except Exception as e:
        raise RuntimeError(f"Error: {str(e)}")