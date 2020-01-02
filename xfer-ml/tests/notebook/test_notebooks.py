import pytest

import os
import subprocess

import nbformat


demo_path = 'docs/demos/'

hpo_notebook = [filename for filename in os.listdir(demo_path) if (filename.endswith('.ipynb') and 'hpo' in filename)]
gluon_notebook = [fname for fname in os.listdir(demo_path) if (fname.endswith('.ipynb') and 'gluon' in fname)]
# Exclude other notebooks because these will be tested separately
demo_notebooks = [filename for filename in os.listdir(demo_path)
                  if (filename.endswith('.ipynb') and filename not in hpo_notebook and filename not in gluon_notebook)]

original_path = os.getcwd()

temp_notebook = 'temp_notebook.ipynb'


def _notebook_run(notebook):
    """
    Execute a notebook via nbconvert and collect output.
       :return: (parsed nb object, execution errors)
    """

    if os.path.isfile(temp_notebook):
        os.remove(temp_notebook)

    with open(temp_notebook, 'w') as fout:
        # with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors",
                "--ExecutePreprocessor.timeout=-1",
                "--output", fout.name, notebook]
        try:
            subprocess.check_call(args)
        except subprocess.CalledProcessError as e:
            if e.returncode == 1:
                # print the message and ignore error with code 1 as this indicates there were errors in the notebook
                print(e.output)
                pass
            else:
                # all other codes indicate some other problem, rethrow
                raise

    with open(temp_notebook, 'r') as fout:
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [output for cell in nb.cells if "outputs" in cell for output in cell["outputs"]
              if output.output_type == "error"]

    os.remove(temp_notebook)
    return nb, errors


def _test_ipynb(notebook):
    # Skip any temporary notebook
    if notebook == temp_notebook:
        return
    # Change working dir to demo directory
    if demo_path is not '':
        os.chdir(demo_path)
    # Run notebook and collect errors
    nb, errors = _notebook_run(notebook)
    # Revert to orginal working directory
    os.chdir(original_path)
    # Assert no errors were collected from notebook
    assert errors == [], 'Errors found in {}\n{}'.format(notebook, errors)


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", demo_notebooks)
def test_ipynb(notebook):
    _test_ipynb(notebook)


@pytest.mark.notebook_hpo
@pytest.mark.parametrize("notebook", hpo_notebook)
def test_hpo_ipynb(notebook):
    _test_ipynb(notebook)


@pytest.mark.notebook_gluon
@pytest.mark.parametrize("notebook", gluon_notebook)
def test_gluon_ipynb(notebook):
    _test_ipynb(notebook)
