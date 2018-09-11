import pytest

import os
import subprocess

import nbformat


demo_path = 'docs/demos/'
demo_notebooks = [filename for filename in os.listdir(demo_path) if filename.endswith('.ipynb')]

if demo_path is not '':
    os.chdir(demo_path)


def _notebook_run(notebook):
    """
    Execute a notebook via nbconvert and collect output.
       :return: (parsed nb object, execution errors)
    """

    temp_notebook = 'temp_notebook.ipynb'
    with open(temp_notebook, 'w') as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
                "--ExecutePreprocessor.timeout=-1",
                "--output", fout.name, notebook]
        subprocess.check_call(args)

    with open(temp_notebook, 'r') as fout:
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell for output in cell["outputs"]
              if output.output_type == "error"]

    os.remove(temp_notebook)
    return nb, errors


@pytest.mark.parametrize("notebook", demo_notebooks)
def test_ipynb(notebook):
    nb, errors = _notebook_run(notebook)
    assert errors == [], 'Errors found in {}'.format(notebook)
