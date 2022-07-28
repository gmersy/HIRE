# HIRE
Hierarchical Residual Encoding for Multiresolution Compression

Data set files on google drive: https://drive.google.com/drive/folders/1gxU9GskX9f60meHUnwcaqs0FvE75tuEN?usp=sharing

Requirements:
- Install the dependencies via `pip install -r requirements.txt`
- We used TRC as our downstream compression as explained in the paper. It can be installed through the link: https://github.com/powturbo/Turbo-Range-Coder 
## Our method: HIRE
- Located in `hier.py`
- `HierarchicalSketch` class implements univariate HIRE
- `MultivariateHierarchical` class implements multivariate hire

Detailed comments explaining the correspondence beteween parts of the code and the paper can be found throughout.  
## Main experiment file
1) Navigate to `experiments.py` and choose baselines you would like to run in `initialize(...)`
2) Change data path to the dataset location
3) Change results path
4) Run file
