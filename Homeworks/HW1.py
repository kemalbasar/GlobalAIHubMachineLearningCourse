1):How u define Machine Learning ?
Answer1:

        Machine learning is a decision-maker system that uses mathematical models for contributing to softwares or software-based machines
through learning from datasets , mostly without human interactions.

2) What are the differences between Supervised and Unsupervised Learning? Specify example 3 algorithms for each of these
Answer2:

        Supervised learning is a machine-learning system with supirvision of human.Linear-Regression , Logistic Regression , Decision Trees
are common supervised learning algorithms.

        Unsupervised learning is a machine-learning system without supirvision of human. K-Means Clustering Algorithm , Apriori algorithm,
Neural Networks ( can be either supervised or unsupervised) are common unsupervised learning algorithms.

3)What are the test and validation set, and why would you want to use them?
Answer3:

        Validation set is data set which you put it in your machine learnin model after train set for adjusting your coefficient of features parameters.
Test set is a data set which used to provide unbiased evaluation of a final model. It is used after train and validation set. This data set
will be encountered from machine learning algorithm first time. This predictions given from testset is the cruical , because it will reveal
how succesfull our created models is.

4) What are the main preprocessing steps? Explain them in detail. Why we need to prepare our data?
Answer4:

        First of all we need to define our business question very well , then all relevent data about to subject is needed to be imported.Then
data has to be processed before fit in a machine learning model.
First step is omitting duplicate datas. In the most of the cases we deal with at machine learning , datas dont gives us an absolute results , and also
we got more than two features . Lets assume that  we got 1000 sample with features X,Y and Z , and we have 3 classes named A,B and C. We got 400 sample
from class A  and 180 samples from it  has  feature X > 50 , and 220 samples from it  got X< 50 , so that our model tends to classified sample as an A
when it comes to feature  X<50. However, bad news here  there is duplicated 25 sample of A which feature X<50. If we had omited that duplicates, we
would be prevented our model of making mistake. So we need to cleare our data set from duplicate values all the time.

        Secondly , imbalanced data is really important part of preprocessing steps .When  we work with data sets which has inbalanced number of quantities
of classes , our model will learn about one class and less about the minor ones. If we are working with  classification problems , because of
our model can not learn much about minor classes , it will couse a lot of false positive when its not the case of major class. There some other
cases with imbalanced data , so that it should be done to do oversampling or undersampling before modelling process.

        Third step of prprocessing data is , missing data problem. Null variables will couse a lot when it come to programming phase. One or more futures
of data can be misssing . It can fill with  median or mod of the values , or that datas can be omitted as well bu its not recommenedn because there is
always demand of a data in machine learning.

        4thly , detection of outliers got some help for optimizing data set.Outlier data is sample of data  which dont show same features from most
of the samples from same class. This can be mistaken data or not . Both can affect algorihtms decision mechanisim.

        Feature scaling of data set is another preprocessing step . It makes learning of model faster with optimizing data into smaller scales.

        Data bining is a data processing step which  reduce effects of observetion errors with  categorizing data into intervals.

        Feature encoding is data  processing method whic applied on a categorical data.This is a method which makes meaning of features understanable
for computer softwares with numarizing it.

5) How you can explore and analyse countionus and discrete variables?
Answer 5:

        If samples have categorical features , like thirt sizes of s,m and l  ,or genders of living ; male  and female , it means there is dicrete
variables . This values can be explored by checking its countable or not.  Discrete variables can  be anaized with mapping methods like one-hot-encoding
for nominal values which is one-to-one mapping of  meaning of objects . For ordinal values its easier to map like 0 for small , 1 for medium and 2 for large
tshirt.

	If samples can  not be countable , then it means it has countionus variables. Countionus variables are harder to deal with compare to discrete variables.
Theres common methods for figuring it out. Most comman method is scaling techniques , scaling techniques put variables into smaller, pre-defined intervals. It makes
them easier to read. There are two mostly use statistical techniques that normalizaton and standartization.In addition continues variables can be analize with
binninq technique which is analizing variables in pre-defined intervals.

6) Variable is discrete variable , at back histogram plot that data sorted into bins . Another graph in the plot is line graph. Kernel density estiamation used to draw that line.
This is a continues representation for the discrete histograms. Data represented by continues probablity densit curve.We can apply label encoding as a preprocessing data.
