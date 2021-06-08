# Category learning & visual areas: Data, analysis and figures

This repository contains code needed to reproduce figure panels as in the manuscript "Mouse visual cortex areas represent perceptual and semantic features of learned visual categories" by Pieter M. Goltstein, Sandra Reinert, Tobias Bonhoeffer and Mark Hübener (Max Planck Institute of Neurobiology). A detailed instruction can be found below. In case of any questions, please do not hesitate to contact us.

---

### Part 0: Setting up the analysis environment

#### Part 0a: Install programs/tools

1. Download and install Python 3.7.10 (Anaconda distribution). Follow instructions on https://anaconda.org
2. Download and install “Git” (this is optional, the code can also be downloaded manually). Follow instructions on https://git-scm.com
3. Download and install the “GIN command line client” (this is optional, the data can also be downloaded manually). Follow instructions on https://gin.g-node.org

#### Part 0b: Download the data and code

1. Open up any type of command line window and change the current directory to a drive/folder in which you like to have the entire project
2. Download data

``` gin get pgoltstein/category-learning-visual-areas ```

3. CD into the repo-folder (called “category-learning-visual-areas”)
4. Download the code (this will be placed in a newly created subfolder “code”)

``` git clone https://github.com/pgoltstein/category-learning-visual-areas.git code ```

Your directory structure should look like this:

```
- Base directory (e.g. category-learning-visual-areas)
  - code
    - ... (code repository contents)
  - data
    - ... (data repository contents)
  - figureout
```

#### Part 0c: Create analysis environment

1. Create the python environment from the yaml file in the code folder

``` conda env create -f ./code/environment.yaml --name catvisareas ```

2. Activate the environment

``` conda activate catvisareas ```

3. All code should be run from the respective "code path", that is, cd into the code directory and run it using python. For example:
```
cd code
cd p1_behavioralchambers
python behavior_chamber_learningcurve.py
```

---

### Part 1: Behavioral chambers

* Data path: “./data/p1_behavioralchambers”
* Code path: “./code/p1_behavioralchambers”

Figure 1
* 1b: behavior_chamber_learningcurve.py
*	1c: performance_per_categorylevel.py
*	1d: performance_on_first_trials.py
*	1e: performance_per_categorylevel.py

Extended Data Figure 1
*	1c: performance_per_categorylevel.py
*	1e,f: boundary_stability.py

---

### Part 2

#### Part 2a: Headfixed behavior

* Data path: “./data/p2a_headfixbehavior”
* Code path: “./code/p2a_headfixbehavior”

Figure 2
*	2b: performance_per_categorylevel.py

Extended Data Figure 2
*	2c: headfix_learningcurve.py
*	2d,e: performance_per_categorylevel.py

#### Part 2b: Retinotopic shift of stimulus

* Data path: “./data/p2b_retinotopybehavior”
* Code path: “./code/p2b_retinotopybehavior”

Figure 2
*	2c-g: retinotopy-analysis-performance.py

Extended Data Figure 2
*	2h-i: retinotopy-analysis-eyetracking.py
