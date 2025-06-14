import os
import yaml
import numpy as np

def save_config_to_file(config, task_type, base_dir=""):
    """
    Saves the config dict as a YAML file in base_dir/configs, with a unique filename.
    """
    configs_dir = os.path.join(base_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    base_filename = f"{task_type}_config.yaml"
    config_path = os.path.join(configs_dir, base_filename)
    counter = 1
    while os.path.exists(config_path):
        config_path = os.path.join(configs_dir, f"{task_type}_config_{counter}.yaml")
        counter += 1

    with open(config_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Config saved to {config_path}")

def parse_llm_yaml(reply):
    """
    Cleans and parses an LLM YAML string, removing code block markers if present.
    Returns a Python dict if parsing succeeds, otherwise returns the raw string.
    """
    try:
        # Remove code block markers if present
        if reply.startswith("```"):
            reply = reply.strip().split('\n', 1)[1]
            if reply.startswith("yaml"):
                reply = reply[4:]
            reply = reply.rsplit('```', 1)[0].strip()
        parsed_config = yaml.safe_load(reply)
        return parsed_config
    except yaml.YAMLError:
        print("Could not parse YAML, returning raw string.")
        return reply
    
def generate_random_array(size=3, dtype='float', min_value=-1, max_value=1):
    """
    Generates a single array of random numbers.
    Args:
        size (int): Number of elements in the array.
        dtype (str): 'float' for float values, 'int' for integer values.
        min_value (numeric): Minimum value (inclusive).
        max_value (numeric): Maximum value (exclusive for int, inclusive for float).
    Returns:
        np.ndarray: The generated array.
    """
    if dtype == 'float':
        array = np.random.uniform(min_value, max_value, size)
    elif dtype == 'int':
        array = np.random.randint(min_value, max_value, size)
    else:
        raise ValueError("dtype must be 'float' or 'int'")
    return array