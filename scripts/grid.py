# read yaml file

# then get all top-level keys and their values

# assert that all values are list typed

# then for each key, get the single item values and based on the combination of all, generate the commands like that:

# --set ${key}=training.${value}

# make them copy-pasteable newline separated


def generate_combinations_of_all_values(config):
    from itertools import product

    keys = list(config.keys())
    values_lists = [config[key] for key in keys]
    assert all(isinstance(values, list) for values in values_lists), "All values must be lists"
    values_stringified = [[f"--set training.{key}={value}" for value in values] for key, values in zip(keys, values_lists)]
    combinations = list(product(*values_stringified))
    return combinations
            
if __name__ == "__main__": 
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="Generate command-line arguments from a YAML configuration file.")
    parser.add_argument("base_command", type=str, help="The base command to which the generated arguments will be appended.")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML configuration file.")
    args = parser.parse_args()

    with open(args.yaml_file, 'r') as file:
        config = yaml.safe_load(file)

    all_values = generate_combinations_of_all_values(config)
    for combination in all_values:
        command = args.base_command + " " + " ".join(combination)
        print(command)