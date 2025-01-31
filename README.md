# Computer Vision for TCV
## Setup
I recomend using a unix-like environment
(either Linux, Mac, or [WSL](https://code.visualstudio.com/docs/remote/wsl) for Windows) to work on this project so that the commands are consistant.

In the cv folder, create a [virtual environment](https://docs.python.org/3/library/venv.html). This can be done in VS Code by opening the cv folder as a project and clicking View -> Command Pallete... -> `Python: Create Virtual Environment...`

Next, install the requirements listed in `requirements.txt`:
```
pip install -r requirements.txt
```
This will install all of the necessary dependencies for the project. If additional dependencies are added to the requirements, this same command can be used to install the new ones.

If the virtual environment is active in a terminal, it will show `(.venv)` in the terminal. To activate the virtual environment in a terminal run:
```
. .venv/bin/activate
```

## Adding Dependencies
To install additional dependencies manually, use:
```
pip install package_name
```
Before you push your changes, make sure to update the `requirements.txt` using:
```
pip freeze > requirements.txt
```