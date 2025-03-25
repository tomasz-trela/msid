# MSID
## Installation

### Prerequisites
Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system.
Download [data](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) and put file in project at path.
```bash
/data/obesity_data.csv
```

### Setting Up the Environment
Run the following commands to set up the project:

```bash
# Create a new Conda environment
conda create --name msid python=3.12.0

# Activate the environment
conda activate msid

# Install dependencies from requirements.txt
pip install -r requirements.txt
```

## Usage
### Running Jupyter Notebook
To start a Jupyter Notebook session:
```bash
jupyter notebook
```
or use vscode [extension for notebook](https://code.visualstudio.com/docs/datascience/jupyter-notebooks)

## License
This project is licensed under the MIT License.
