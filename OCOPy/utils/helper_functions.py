# utils/helper_functions.py

def load_data(file_path):
    """Load data from a file."""
    # Example function, replace with actual implementation
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    return data

def preprocess_data(data):
    """Preprocess data."""
    # Example function, replace with actual implementation
    processed_data = [item.lower() for item in data]
    return processed_data
