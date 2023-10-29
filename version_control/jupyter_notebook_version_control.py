import subprocess
import argparse

def remove_extension(filename):
    # Find the last occurrence of '.' to handle cases where the filename itself contains dots
    last_dot_index = filename.rfind('.')
    
    # If a dot is found, remove the extension; otherwise, return the original filename
    if last_dot_index != -1:
        return filename[:last_dot_index]
    else:
        return filename

def convert_notebook(notebook_filename, output_format):
    subprocess.run(['jupyter', 'nbconvert', '--to', output_format, notebook_filename+'.ipynb' , '--output-dir="./version_control2"', '--output', notebook_filename])
    print(f"File '{notebook_filename}.{output_format}' created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Jupyter Notebook to different formats.')
    parser.add_argument('notebook_file', help='Path to the Jupyter Notebook file')
    parser.add_argument('--output-format', '-o', default='html', help='Output format (e.g., html, pdf)')
    args = parser.parse_args()

    notebook_file = remove_extension(args.notebook_file)

    convert_notebook(notebook_file, args.output_format)
