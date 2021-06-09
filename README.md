# Category learning & visual areas: Data, analysis and figures

This repository contains code needed to reproduce figure panels as in the manuscript "Mouse visual cortex areas represent perceptual and semantic features of learned visual categories" by Pieter M. Goltstein, Sandra Reinert, Tobias Bonhoeffer and Mark Hübener (Max Planck Institute of Neurobiology).  
A detailed instruction can be found below. In case of any questions, please do not hesitate to contact us.

---

### Part 0: Setting up the analysis environment

#### Part 0a: Install programs/tools

1. Download and install Python 3.7.10 (Anaconda distribution).  
_Follow instructions on https://anaconda.org_
2. Download and install “Git” (this is optional, the code can also be downloaded manually).  
_Follow instructions on https://git-scm.com_
3. Download and install the “GIN command line client” (this is optional, the data can also be downloaded manually).  
_Follow instructions on https://gin.g-node.org_

#### Part 0b: Download the data and code

1. Open up any type of command line window and change the current directory to a drive/folder in which you like to have the entire project
2. Download data  
``` gin get pgoltstein/category-learning-visual-areas ```

3. CD into the repo-folder (called “category-learning-visual-areas”)
4. Download the code (this will be placed in a newly created subfolder “code”)  
``` git clone https://github.com/pgoltstein/category-learning-visual-areas.git code ```

Your directory structure should look like this:
```
- category-learning-visual-areas (or any other name you chose for your base directory)
  - code
    - p1_behavioralchambers
      - behavior_chamber_learningcurve.py
      - ... etc
    - p2a_headfixbehavior
    - ... etc
  - data
    - chronicrecordings
    - p1_behavioralchambers
    - ...
  - figureout
  - ... etc
```

#### Part 0c: Create analysis environment

1. Create the python environment from the yaml file in the code folder  
``` conda env create -f ./code/environment.yaml --name catvisareas ```

2. Activate the environment  
```conda activate catvisareas ```

3. All code should be run from the respective "code path", that is, cd into the code directory and run it using python. This is for the reason that it (by defaut) will look for the data in a reletive path starting from the folder where the python code is stored. So, for example, to make a learning curve as in figure 1b:
```
cd code
cd p1_behavioralchambers
python behavior_chamber_learningcurve.py
```

---

### Part 1: Behavioral chambers

Data path: “./data/p1_behavioralchambers”  
Code path: “./code/p1_behavioralchambers”

Figure 1
* 1b: ``` python behavior_chamber_learningcurve.py ```
*	1c: ``` python performance_per_categorylevel.py ```
*	1d: ``` python performance_on_first_trials.py ```
*	1e: ``` python performance_per_categorylevel.py ```

Extended Data Figure 1
*	ED1c: ``` python performance_per_categorylevel.py ```
*	ED1e,f: ``` python boundary_stability.py ```

---

### Part 2a: Headfixed behavior

Data path: “./data/p2a_headfixbehavior”  
Code path: “./code/p2a_headfixbehavior”

Figure 2
*	2b: ``` python performance_per_categorylevel.py ```

Extended Data Figure 2
*	ED2c: ``` python headfix_learningcurve.py ```
*	ED2d,e: ``` python performance_per_categorylevel.py ```

---

### Part 2b: Retinotopic shift of stimulus

Data path: “./data/p2b_retinotopybehavior”  
Code path: “./code/p2b_retinotopybehavior”

Figure 2
*	2c-g: ``` python retinotopy-analysis-performance.py ```

Extended Data Figure 2
*	ED2h-i: ``` python retinotopy-analysis-eyetracking.py ```

---

### Part 3a: Chronic imaging behavior

Data path: “./data/p3a_chronicimagingbehavior”  
Code path: “./code/p3a_chronicimagingbehavior”

Figure 3
* 3d,e: ``` python performance_per_categorylevel.py category ```
* 3e: ``` python performance_per_categorylevel.py baseline ```

Extended Data Figure 4
* ED4a: ``` python chronicimagingbehavior_learningcurve.py ```
* ED4b: ``` python performance_per_categorylevel.py category ```

---

### Part 3b: Cortical inactivation

Data path: “./data/p3b_corticalinactivation”  
Code path: “./code/p3b_corticalinactivation”

Figure 3
* 3f: ``` python corticalinactivation.py ```

Extended Data Figure 3
* ED4c,d: ``` python corticalinactivation.py ```

---

### Part 3c: Visual area inactivation

Data path: “./data/p3c_visualareainactivation”  
Code path: “./code/p3c_visualareainactivation”

Extended Data Figure 5
* ED5d: ``` python visualareainactivation-analysis.py ```

---

### Part 4: Fraction responsive neurons

Data path: “./data/chronicrecordings”  
Processed data path: “./data/p4_fractionresponsiveneurons”  
Code path: “./code/p4_fractionresponsiveneurons”

Data pre-processing:
* ``` python frac-n-resp-cat-subsampl.py [areaname] ```  
Calculates subsampled fraction of responsive neurons for each visual area.

Figure 3
* 4a-f: ``` python fractionresponsiveneurons-clustering.py ```
* 4g-i: ``` python fractionresponsiveneurons-linearmodel.py ```

Extended Data Figure 6
* ED6a: ``` python fractionresponsiveneurons-clustering.py ```
* ED6b: ``` python fractionresponsiveneurons-linearmodel.py ```

---

### Part 5: Encodingmodel analysis

Data path: “./data/chronicrecordings”  
Processed data path: “./data/p5_ encodingmodel”  
Code path: “./code/p5_ encodingmodel”  

Data pre-processing:
* ``` python run-encodingmodel-within.py [areaname] [mousename] comb -o R2m -r trials -l Category -c Trained ```  
Fits the full encoding model to the activity trace of each neuron in a singe chronic recording (identified by area and mouse).

* ``` python run-encodingmodel-within.py [areaname] [mousename] comb -o R2m -r trials -l Category -c Trained -s trials ```  
Fits the full encoding model to the shuffled activity trace of each neuron in a singe chronic recording (identified by area and mouse).

* ``` python run-encodingmodel-delta.py [areaname] [mousename] comb -o R2m -l Category -c Trained -g [group selector] ```  
Fits a full encoding model with one regressor group shuffled to the activity trace of each neuron in a singe chronic recording (identified by area and mouse).

* ``` python run-encodingmodel-delta.py [areaname] [mousename] comb -o R2m -l Category -c Trained -a [group selector] ```  
Fits a full encoding model with all but one regressor group shuffled to the activity trace of each neuron in a singe chronic recording (identified by area and mouse).

* ``` python run-encodingmodel-regularization.py [areaname] [mousename] ```  
Repeatedly fits a full encoding model to the activity trace of each neuron in a singe chronic recording (identified by area and mouse) using different L1 values.

Figure 5
* 5f: ``` python encodingmodel-full-area-dorsal-ventral.py ```
* 5g: ``` python encodingmodel-delta-component-area.py ```

Extended Data Figure 7
* ED7a: ``` python encodingmodel-full-regularization.py ```
* ED7b,c: ``` python encodingmodel-full-R2-fraction.py ```
* ED7d: ``` python encodingmodel-full-vs-responsivefraction.py ```
* ED7e,f: ``` python encodingmodel-full-area-dorsal-ventral.py ```
* ED7g: ``` python encodingmodel-delta-component-area.py ```

---

### Part 6a: Encodingmodel category tuning

Data path: “./data/chronicrecordings”  
Processed data path: “./data/p5_ encodingmodel”  
Code path: “./code/p6a_ encmodelcategorytuning”  

Data pre-processing: see part 5

Figure 6
* 6b-f: ``` python encmodel-cti-semantic-feature.py ```
* 6h,i: ``` python encmodel-deltacti-vs-choice.py ```

Extended Data Figure 8
* ED8b,c: ``` python encmodel-kernelframes.py ```

Extended Data Figure 9
* ED9a-c: ``` python encmodel-cti-semantic-feature.py ```

Extended Data Figure 10
* ED10a,b: ``` python encmodel-psth.py ```
* ED10c-e: ``` python encmodel-tuningproperties.py ```

Note: The script encmodel-cti-semantic-feature.py plots many individual data points and links them using lines, which runs generally very slow. In the script is an option for not showing the individual datapoints, the variable is called "suppress_connecting_individual_datapoints_for_speed" (on line 34), set this to "True" and the script will run much faster, but not plot some of those individual datapoints.
