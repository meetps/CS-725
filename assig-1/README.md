Meet Pragnesh Shah
13D070003


### Code : 

`main.py` contains the code for the entire regression problem.


### Technique


#### Model and Feature Engineering

* Ridge Regression with weighing parameter alpha=0.5
* 10/5-fold Cross Validation from train data

#### Preprocessing

* Normalization
* Outlier removal by keeping entires in (mean +/- ratio * std)


#### PostProcessing

Rounding of to nearest hundreds as all data in train output is in hundreds.



### Run Code 

`python main.py`

### Output

`output` directory.


### Additional

I also tried to train a neural network on the data and got  a pretty good score as well.
The neural network can be found in the `train.py` file.
