# deepquantization

#Installation

First of all, set up a conda environment
```
conda create --name deepconcolic python==3.7
conda activate deepconcolic
```
This should be followed by installing software dependencies:
```
conda install opencv nltk matplotlib
conda install -c pytorch torchvision
pip3 install numpy==1.19.5 scipy==1.4.1 tensorflow\>=2.4 pomegranate==0.14 scikit-learn scikit-image pulp keract np_utils adversarial-robustness-toolbox parse tabulate pysmt saxpy keras menpo patool z3-solver pyvis
```

To run the tests first you must specify the parameters: dataset (cifar10, mnist, cats_and_dogs), coverage (ssc, nc), generator (fuzzing, cfg, fuzzing_cfg), quantize.

As an example to run one of the complete test, which were used in the dissertation project use the command:
```
python main.py --dataset cats_and_dogs --coverage ssc --generator fuzzing_cfg --quantize
```

