# Alpha-NonZero

<p align="center"><img src="assets/aichess.png" width="50%"></p>

## Install

##### Clone the project from GitHub
```bash
git clone https://github.com/jorgenwh/alpha-nonzero.git
cd alpha-nonzero
```

##### Fetch submodules
```bash
git submodule update --init --recursive
```

##### Set up a Python virtual environment and update
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

##### Install necessary third-party dependencies
```bash
pip install -r requirements.txt
```

##### Install pystockfish
```bash
cd pystockfish
pip install .
cd ..
```

##### Install alpha-nonzero
```bash
pip install -e .
```

You should now be able to run all the scripts in the 'scripts' directory :)
