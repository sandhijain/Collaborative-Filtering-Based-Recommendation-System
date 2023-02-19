import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
st.title("Bangalore Zomato Data")

#import our data
data = pd.read_csv('BangaloreZomatoData.csv')
#st.write(data.head())

# Drop columns from the DataFrame
columns_to_drop = ['Timing','Full_Address','PhoneNumber','isTakeaway','KnownFor','PopularDishes','PeopleKnownFor']
rename_cols = data.rename(columns={'IsHomeDelivery':'Home Delivery','isIndoorSeating':'Indoor Seating','isVegOnly':'Vegetarian','AverageCost':'Average Cost', 'Area': 'Areax'})
df = rename_cols.drop(columns_to_drop, axis=1) 
df['Dinner Ratings'] = pd.to_numeric(df['Dinner Ratings'], errors='coerce')
df['Delivery Ratings'] = pd.to_numeric(df['Delivery Ratings'], errors='coerce')

  
df[['Area', 'Area1', 'Area2']] = df['Areax'].str.split(',', expand=True)
columns_to_drop1 = ['Areax','Area1','Area2']
df =  df.drop(columns_to_drop1, axis=1) 
# st.write("This is the updated table")
# st.write(df.head())

#lable encoding ratings column

lab_enc = preprocessing.LabelEncoder()
Y = df['Dinner Ratings']
df_transformed = df
# rename_cols1 = df_transformed.rename(columns={'Area0':'Area'})
# df_transformed = rename_cols1.drop(columns_to_drop, axis=1)


Y_encoded = lab_enc.fit_transform(Y)
Y_encoded = pd.DataFrame({'rating_encoded':Y_encoded})
df_transformed = pd.concat([df_transformed, Y_encoded], axis=1)



 
#APPLYING KNN
#df_transformed =df_transformed.insert(1, 'rest_id', range(1, 1 + len(df_transformed)))
df_transformed.insert(1, 'rest_id', range(1, 1 + len(df_transformed)))
df = df_transformed
#st.write(df.head())

#MODEL TRAINING

#1) recommend_rating:
#This function implements knn and recommends restaurants based on rating of the input restaurant
def recommend_rating(inp_rest_id, num_of_recommendation):
    X = df[['rest_id']]
    Y = df['rating_encoded']
    knc = NearestNeighbors(metric='cosine', n_neighbors=num_of_recommendation, algorithm='brute', n_jobs=-1)
    knc.fit(X, Y)
    neighbor_distances, knc_neigbors = knc.kneighbors([[inp_rest_id]])
    knc_neigbors = knc_neigbors[0]
    st.write(f"\n{num_of_recommendation} neighbors: {knc_neigbors}\n")
    st.write(f"Distance of neighbors: {neighbor_distances[0]}\n")
    recommended_rest = pd.DataFrame()
    for item in knc_neigbors:
        rest_entry = df.loc[df['rest_id'] == item]
        recommended_rest = recommended_rest.append([rest_entry])
    recommended_rest = recommended_rest.sort_values('Dinner Ratings', ascending=False)
    return recommended_rest 

#2) recommend_online:
#This function implements knn and recommends restaurants which have online delivery option available based on rating of the input retaurant 

def recommend_online(inp_rest_id, num_of_recommendation):
    rest_online = df[(df['Home Delivery'] == 1)]
    X = rest_online[['rest_id']]
    Y = rest_online['rating_encoded']
  

    knc = KNeighborsClassifier(metric = 'cosine', n_neighbors=num_of_recommendation, algorithm='brute', n_jobs=-1)
    knc.fit(X,Y)
    neighbor_distances, knc_neigbors = knc.kneighbors([[inp_rest_id]])
    knc_neigbors = knc_neigbors[0]
    print("\n",num_of_recommendation,"neighbors: ", knc_neigbors)
    print("\nDistance of neighbors: ", neighbor_distances[0], "\n")
  
    recommended_rest = pd.DataFrame()
    for item in knc_neigbors:
        rest_entry = rest_online.loc[rest_online['rest_id'] == item]
        recommended_rest=recommended_rest.append( [rest_entry] )
        recommended_rest = recommended_rest.sort_values('Dinner Ratings', ascending=False)

    return recommended_rest    #3) recommend_according_to_budget:
#This function implements knn and recommends restaurants based on rating of the input retaurant and filters out the restaurants that are outside the budget provided by the user

def recommend_according_to_budget(inp_rest_id, num_of_recommendation, upper_limit):
    rest_budget = df[(df['Average Cost'] <= upper_limit)]
    X = rest_budget[['rest_id']]
    Y = rest_budget['rating_encoded']
     

    knc = KNeighborsClassifier(metric = 'cosine', n_neighbors=num_of_recommendation, algorithm='brute', n_jobs=-1)
    knc.fit(X,Y)
    neighbor_distances, knc_neigbors = knc.kneighbors([[inp_rest_id]])
    knc_neigbors = knc_neigbors[0]
    print("\n",num_of_recommendation,"neighbors: ", knc_neigbors)
    print("\nDistance of neighbors: ", neighbor_distances[0], "\n")

    recommended_rest = pd.DataFrame()
    for item in knc_neigbors:
        rest_entry = rest_budget.loc[rest_budget['rest_id'] == item]
        recommended_rest=recommended_rest.append( [rest_entry] )
        recommended_rest = recommended_rest.sort_values('Dinner Ratings', ascending=False)


    return recommended_rest

#4) recommend_according_to_budget_and_online:
#This function implements knn and recommends restaurants which have online delivery option available based on rating of the input retaurant and 
#filters out the restaurants that are outside the budget provided by the user

def recommend_according_to_budget_and_online(inp_rest_id, num_of_reccomendation, upper_limit):
    rest_online = df[(df['Home Delivery'] == 1)]
    rest_budget = rest_online[(rest_online['Average Cost'] <= upper_limit)]
    X = rest_budget[['rest_id']]
    Y = rest_budget['rating_encoded']

  # euclidean , cosine
    knc = NearestNeighbors(metric = 'cosine', n_neighbors=num_of_reccomendation, n_jobs=-1)
    knc.fit(X,Y)
    neighbor_distances, knc_neigbors = knc.kneighbors([[inp_rest_id]], n_neighbors=num_of_reccomendation)
    knc_neigbors = knc_neigbors[0]
    print("\n",num_of_reccomendation,"neighbors: ", knc_neigbors)
    print("\nDistance of neighbors: ", neighbor_distances[0], "\n")

    recommended_rest = pd.DataFrame()
    for item in knc_neigbors:
        rest_entry = rest_budget.loc[rest_budget['rest_id'] == item]
        recommended_rest=recommended_rest.append( [rest_entry] )
        recommended_rest = recommended_rest.sort_values('Dinner Ratings', ascending=False)

    return recommended_rest 


st.markdown(
    '''
    <style>
    .sidebar .sidebar-content {
        background-color: #E6E6FA;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 0 1rem 0 rgba(0, 0, 0, 0.2);
    }
    </style>
    ''',
    unsafe_allow_html=True,
)



# Show a sidebar with input fields
with st.sidebar:
    st.write("## Input")
    inp_rest_id = st.number_input("Enter the restaurant ID: [1,8923]", min_value=1, value=1)
    num_of_recommendation = st.number_input("Enter number of clusters you want for the restaurants", min_value=1, max_value=7, value=5)
    online_available = st.selectbox("Should the restaurant have Online order facility", ["Yes", "No"])
    budget = st.selectbox("Filter according to budget?", ["Yes", "No"])
    if budget == "Yes":
        upper_limit = st.number_input("Enter the budget. Upper limit", min_value=0, value=500)

# Use the input values to generate recommendations
inp_rest = df_transformed[(df_transformed['rest_id'] == inp_rest_id)]
st.write("Details of input restaurant:")
st.write(inp_rest[['Name', 'rest_id', 'Dinner Ratings', 'rating_encoded', 'Area']])

if (online_available == "Yes") & (budget == "Yes"):
    recommended_rest = recommend_according_to_budget_and_online(inp_rest_id, num_of_recommendation, upper_limit)
elif (online_available != "Yes") & (budget == "Yes"):
    recommended_rest = recommend_according_to_budget(inp_rest_id, num_of_recommendation, upper_limit)
elif (online_available == "Yes") & (budget != "Yes"):
    recommended_rest = recommend_online(inp_rest_id, num_of_recommendation)
elif (online_available != "Yes") & (budget != "Yes"):
    recommended_rest = recommend_rating(inp_rest_id, num_of_recommendation)

button_clicked = st.sidebar.button('Enter')
if button_clicked:    
    st.write("Recommended restaurants:")
    st.write(recommended_rest[['Name', 'rest_id', 'Dinner Ratings', 'rating_encoded', 'Area', 'Cuisines', 'Average Cost']])

 
