from nbconvert import PythonExporter
import nbformat
import os

def convert_ipynb_to_python(ipynb_file, output_folder):
    # Load the notebook
    with open(ipynb_file, 'r', encoding='utf-8') as nb_file:
        notebook = nbformat.read(nb_file, as_version=4)

    # Create a Python exporter
    exporter = PythonExporter()

    # Convert the notebook to a Python script
    (python_script, resources) = exporter.from_notebook_node(notebook)

    # Define the output file path
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(ipynb_file))[0] + '.py')

    # Write the Python script to the output file
    with open(output_file, 'w', encoding='utf-8') as py_file:
        py_file.write(python_script)

    print(f"Conversion successful: {ipynb_file} -> {output_file}")

def batch_convert_ipynb_to_python(input_txt, input_folder, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Read the list of IPython notebooks from the text file
    with open(input_txt, 'r', encoding='utf-8') as file:
        ipynb_list = [line.strip() for line in file]

    # Convert each IPython notebook in the list
    for ipynb_file in ipynb_list:
        # Construct the full file path with the input folder
        full_path = os.path.join(input_folder, ipynb_file)
        convert_ipynb_to_python(full_path, output_folder)

# Example usage
list_of_files = "list_of_jupyter_notebooks_to_be_converted_to_py.txt"  # Update with your file containing the list of notebook filenames
jupyter_notebooks_folder = "./.."  # Update with the path to the directory containing the notebooks
output_folder_path = "./output_python_scripts"

batch_convert_ipynb_to_python(list_of_files, jupyter_notebooks_folder, output_folder_path)
