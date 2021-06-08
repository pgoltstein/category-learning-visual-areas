# Category learning & visual areas: Data, analysis and figures

_Pieter Goltstein_

### Part 0a: Install analysis environment

* Download and install Python 3.7.10 (Anaconda distribution). Follow instructions on https://anaconda.org
* Download and install “Git” (this is optional, the code can also be downloaded manually). Follow instructions on https://git-scm.com
* Download and install the “GIN command line client” (this is optional, the data can also be downloaded manually). Follow instructions on https://gin.g-node.org

### Part 0b: Download the data and code

* Open up any type of command line window and change the current directory to a drive/folder in which you like to have the entire project
* Download data

``` gin get pgoltstein/category-learning-visual-areas ```

* CD into the repo-folder (called “category-learning-visual-areas”)
* Download the code (this will be placed in a newly created subfolder “code”)

``` git clone https://github.com/pgoltstein/category-learning-visual-areas.git code ```

* Your directory structure should look like this

```
- Base directory (e.g. category-learning-visual-areas)
  - code
    - ... (code repository contents)
  - data
    - ... (data repository contents)
  - figureout
```

### Part 0c: Create analysis environment

* Create the python environment from the yaml file in the code folder

``` conda env create -f ./code/environment.yaml --name catvisareas ```

* Activate the environment

``` conda activate catvisareas ```
