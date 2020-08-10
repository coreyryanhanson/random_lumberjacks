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
                   "tensorflow-jupyter-gpu": "coreyhanson/anacuda-tensorflow",
                   "pytorch-jupyter-gpu": "coreyhanson/anacuda-pytorch"
                  }
    scraping_dict = {"scraping-jupyter-cpu": "jupyter/scipy-notebook",
                     "scraping-jupyter-gpu": "coreyhanson/anacuda-scipy",
                 }

    base_lines = parse_parent("base-Dockerfile")
    scraping_lines = parse_parent("scraping-Dockerfile")
    append_neural_lines = parse_parent("neural-network-append")
    append_jupyter_ext_lines = parse_parent("jupyter_extensions-append")

    regular_doc = base_lines + append_jupyter_ext_lines
    neural_doc = base_lines + append_neural_lines + append_jupyter_ext_lines
    scraping_doc = scraping_lines + append_jupyter_ext_lines

    modify_children(regular_doc, file_dict)
    modify_children(neural_doc, neural_dict)
    modify_children(scraping_lines, scraping_dict)

    print("Files written")

if __name__ == "__main__":
    main()
