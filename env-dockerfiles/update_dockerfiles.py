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
    neural_cpu = {"tensorflow-jupyter-cpu": "jupyter/tensorflow-notebook"}
    neural_gpu = {"tensorflow-jupyter-gpu": "anacuda-tensorflow"}
    
    base_lines = parse_parent("base-Dockerfile")
    neural_lines = parse_parent("neural-network-append")
    
    neural_cpu_lines = base_lines + neural_lines
    neural_gpu_lines = base_lines + modify_lines(neural_lines, "keras", "keras-gpu")
    
    modify_children(neural_cpu_lines, neural_cpu)
    modify_children(neural_gpu_lines, neural_gpu)
    
    print("Files written")

if __name__ == "__main__":
    main()