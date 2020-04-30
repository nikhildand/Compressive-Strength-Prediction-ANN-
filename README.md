# Compressive-Strength-Prediction-ANN

**Author** : Nikhil Dandamudi

**Specialization Addressed** : Structural Engineering (Through Neural Networks)

**Description** : Artificial Neural Networks Have Been A Breakthrough In Many Sectors, The Presented Model Uses ANN To Predict The Compressive Strength Of Concrete With Different Mix Proportions. The Present Model Achieve's A R2 Score Of 0.7137, Which Isn't That Bad

**Note : Since I Have Picked Up This Project Again, I Will Be Modifying My Architecture, Would Re-Train The Model, If Any Of Which Would Result In Better Performance Would Update The New Weights & Architecture. (Also I Might Be Switching From Keras To PyTorch Framework).**

# Algorithm & Architecture: 

1) The Data Set Was Obtained From http://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength, by Prof. I-Cheng Yeh, Your Effort Is Really Appreciated !! 

2) The Data Set Was Divided Into Training And Test Set With The Starting [0:926] Belonging To The Training Set And The Last [926:1030] Belonging To Test Set For Validating Our Results. The Division Was Approximatly Around 90% Train, % 10% Test.

3) Then Feature Normalization Was Done To the Training Data Before It Was Fed Into Our Neural Network.

4) Optional : I Have Implemented A Live Training Visualization Of The Cost Function Variation For Both Training And Test Data, Which I Had Compiled From Another Source From Towards Data Science Platform, Unable To Find The Link To The Article, Will Update When I Have Found It.

5) **Cost Function Used** : Mean Absolute Percentage Error, **Optimizer Used** : Adam (Gradient Descent), With Learning Rate = 0.001, **Batch Size Used** : 32, I Dont Remember The Number Of Epochs I Had To Run It For.

6) It Is Pretty Much An Implementation Of A Dense Neural Network With 2 Hidden Layer, With Each Node Consisting Of 25 Node's (Neurons), With ReLu As An Activation Function For The Hidden Layers, And Linear Activation Function For The Last Layer With One Node Since, It Is A Regression Problem.

![](/Utilities/Sigmoid-ReLU-and-soft-ReLU-functions.png)

7) A Scatter Plot Between Actual Strength To Predictions Of Test Data's Strength Of Concrete Was Plotted On X & Y Axis Respectively. The Closer They Are To The X=Y Line The Better Are The Predictions. (**Current** R2 Score : 0.7137, Which Could Change In Future Updates.)
