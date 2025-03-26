# MSID
## Installation

### Clone repository
```bash
git clone https://github.com/tomasz-trela/msid.git
```

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
conda env create -f environment.yml

# Activate the environment
conda activate msid
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
