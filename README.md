# flask-ui-knn-optimization
Flask interface to find optimal input parameters for K Nearest Neighbor machine learning classifier


# Preface
Machine learning classification requires iterations of different input parameters in order maximize accuracy of a model. This application takes data and loops through different training/testing splits and iterates through up to 30 neighbors in order to plot all of the accuracies. This will limit the coding required to evaluate model performance.

# Requirements

- Basic understanding of K Nearest Neighbor
- Cleaned, pre-processed data to train and test / Or use pre-loaded dummy data
- Basic understanding of the Terminal

# How to Host the Application

Clone the repository locally. 

In the Terminal, navigate to the cloned repo location and run the following commands: 

If virtual env is not installed:

```
pip install virtualenv
```

Start virtual environment:

```
virtualenv virt
```

```
source virt/bin/activate
```

Install requirements: 
```
pip install -r requirements.txt
```

Launch application: 
```
python application.py
```

Once application has been launched, it will be hosted locally on your machine, accessible here. (Performance is best in Chrome)
```
http://0.0.0.0:80/
```

# Using the Application

The application accepts cleaned and labeled CSV uploads of any data sets for training and testing. The data that is uploaded must first be pre-processed. 

Alternatively, there is dummy data stored in the application's memory, which is sourced from Kaggle. It is a dataset using HR analytics to classify employees likely to leave a company. Full dataset can be found here: https://www.kaggle.com/jacksonchou/hr-data-for-analytics
