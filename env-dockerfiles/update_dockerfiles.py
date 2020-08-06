import re

def parse_parent(path):
    with open(path, "r") as f:
        return f.readlines()

def write_child(path, lines):
    with open(path, "w") as f:
        f.writelines(lines)

def modify_children(lines, file_dict):
    for key, value in file_dict.items():
        lines[0] = f"FROM {value}\n"
        write_child(f"{key}/Dockerfile", lines)

def modify_lines(lines, term, replacement):
    return [line.replace(term, replacement) for line in lines]

def main():
    file_dict = {"scipy-jupyter-cpu": "jupyter/scipy-notebook",
                 "scipy-jupyter-gpu": "coreyhanson/anacuda-scipy",
                 "pyspark-jupyter-cpu": "jupyter/pyspark-notebook",
                 "pyspark-jupyter-gpu": "coreyhanson/anacuda-pyspark"
                 }
    neural_dict = {"tensorflow-jupyter-cpu": "jupyter/tensorflow-notebook",
                   "tensorflow-jupyter-gpu": "coreyhanson/anacuda-tensorflow"
                   "pytorch-jupyter-gpu": "coreyhanson/anacuda-pytorch"
                  }

    base_lines = parse_parent("base-Dockerfile")
    appened_neural_lines = parse_parent("neural-network-append")

    neural_lines = base_lines + appened_neural_lines

    modify_children(base_lines, file_dict)
    modify_children(neural_lines, neural_dict)


    print("Files written")

if __name__ == "__main__":
    main()
