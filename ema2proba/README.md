# Maedeep

## Presentation

Maedeep is a python interface to use the articulatory model by Maeda. It allows forward and inverse mapping between the following spaces:
* Articulatory parameters
* Contours
* Area function
* Task variables
* Transfer function
* Formants

Inverse mapping is performed using a DNN model trained on simulated data. The code to compute contour sand area functions is derived from VTsynth toolbox by Satrajit Ghosh [1].

## Installation

Maedeep is implemented with Python3. Tests have been performed with Python 3.9, so consider having a version of Python that is at least as new as Python 3.9.

Please also consider using a Python3 virtual environment during the development stage:
```bash
# create a python 3 virtual environment and activate it
$ python3 -m venv maedeep_env
$ source maedeep_env/bin/activate
```

You can get the source codes by cloning this repo with git
```
git clone https://git.ecdf.ed.ac.uk/belie/maedeep.git

```

PlanArt is not yet on PyPI. You can install it locally in editable mode:
```bash
# install planart locally in editable mode
$ python3 -m pip install -e path_to_maedeep
```

* WARNING: Please, do not modify anything in the planart folder, unless you want to modify the repository.

## Checking the software behavior
To check if everything runs as expected, run the following command:
```bash
# test if everything works as expected
$ cd path_to_maedeep
$ pytest tests
```
It should not return any error message

## Running demos
Some demos are available in the `demos` folder. Note that they require some external dependencies that you might want to install prior to running them. Dependencies are:
* matplotlib
* spychhiker (installing spychhiker using pip will automatically install matplotlib)

## Contact

For any additionnal information, please contact Benjamin Elie at benjamin.elie (at) ed.ac.uk

## License

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.

## References

* [1] Satrajit Ghosh, "VocalTractModels", github: https://github.com/sensein/VocalTractModels

