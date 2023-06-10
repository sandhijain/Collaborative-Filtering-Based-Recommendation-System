# Machine-learning-zomato-data-2022
## Restaurant Recommendation System

Nowadays, all information is driven by the tangled and intricate data humans have collected over the years. With the enlightenment of knowledge it has become important to extract out the important information. For completing this task, Machine Learning, a specialised branch of Artificial Intelligence has come a long path and made way through the prediction of answers without human interference.

A Recommendation System is an application of Machine Learning which is used to recommend similar choices to people based on their previous selections. Recommendation system is categorized in three classes: Collaborative Filtering, Content based and hybrid based Approach. For our Zomato Bangalore Restaurant Data set, we have used Collaborative Filtering with the help of K Nearest Neighbour Algorithm to recommend restaurant based on certain filters like budget, home delivery options and initial choice. 

### INTRODUCTION

A Recommendation System is required in most of the applications for better customer experience. In this project when a customer is searching for a restaurant, the model will help recommend more such restaurants for viewer to explore and have choices to choose from. Before applying the algorithm, we have done an exploratory data analysis to understand and structure the data is a form which is easier to understand and grasp. 

In this project we have used dataset from Kaggle: Zomato Bangalore Restaurants 2022. This system  takes input from the user and provides recommendations based on ratings of the input restaurant Dataset. The Machine Learning algorithm used in this project is KNN (K- nearest neighbors). The user finds nearest neighbors of input restaurant rating. User has the choice to filter the recommendations based on their budget. User can also provide an upper limit of their budget and the system recommends restaurants based on average cost

### K NEAREST NEIGHBOUR ALGORITHM

K Nearest neighbour or KNN is a simple supervised Machine Learning Algorithm which is used in both classification and regression problems. It is very effective but its effectiveness decreases as data size increases. 
KNN works as a unary or m-ary classifier. It predicts or recommends based on the distance approaches. In our project, we have classified restaurants based on different parameters and after taking a single input from the user, the KNN algorithm comes to work to recommend more such restaurants with good ratings. 

### DATA PREPROCESSIONG
We have used the following libraries in this project: 
 pandas, numpy , matplotlib, sklearn, pickle, KNeighborsClassifier and NearestNeighbors 
 

We renamed the columns using the rename function:
'IsHomeDelivery':'Home Delivery'
'isIndoorSeating':'IndoorSeating'
'isVegOnly':'Vegetarian'
'AverageCost':'Average Cost’

Then some unuseful columns were removed:
'Timing','Full_Address','PhoneNumber','isTakeaway','KnownFor','Delivery Reviews','PopularDishes','PeopleKnownFor'
Duplicate columns were also removed.

### EXPLORATORY DATA ANALYSIS

A histogram is for the dataset. To simplify the plots based on Area we split the columns into three columns: Area 0, Area1 and Area2 on the comma. Since Area1 and Area2 mostly contain Bangalore, we do not consider Area. Only those restaurants were picked which had more than 50 orders. We plotted a bar plot between location and number of restaurants and found that most restaurants are in Electronic City.

We plotted a matplotlib plot between number of online /offline orders vs number of orders. This insight showed that mostly all restaurants offer home delivery service.

A box plot is plotted to identify relationship between average cost of restaurants and online vs offline orders. Another box plot is  plotted between average cost vs dinner ratings.

A bar plot is plotted between location and number of offline vs online orders. It is noticed that there are very few restaurants that do not offer home delivery.

A seaborn bar plot is plotted between number of reviews vs cuisines to find the most popular cuisines. We found that Continental, Asian, Pizza, Burger and Biryani is the most popular cusine having approximately 40,000 reviews.


### APPLYING MACHINE LEARNING ALGORITHM
For applying KNN we have inserted a new column ‘rest_id’ . We performed label encoding on the dinner ratings column and then performed model training.
There are four ways in which we recommend restaurants:
1)	On the basis of ratings of input restaurant
2)	On the basis of Home Delivery service  
3)	Based on budget provided by user  
4)	Home Delivery service plus on the basis of budget provided by user.

Finally we created a function to take input from user on the above parameters and recommend restaurants to the user.	This application is deployed on Streamlit.
