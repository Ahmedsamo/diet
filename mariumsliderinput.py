import streamlit as st, pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


st.title('**انتي الأفضل يامريم**')
st.subheader('لنبدأ سويا رحلة النجاح')



# sidebar parameters
#-----------------------------------

st.sidebar.header('User Input Parameters')

def user_input_features():
    weight = st.sidebar.slider('weight', 50, 100, 80)  #4.3 minimum  7.9 max  5.4  default
    #smm = st.sidebar.slider('smm', 10,30 20)
    #Fat_mass = st.sidebar.slider('Fat_mass', 25, 50, 35)
    
    data = {'weight': weight,
            #'smm': smm,
            #'Fat_mass': Fat_mass
             }
           
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)



#من هنا هنبدأندخل البراميترز في الداتا فريم
#------------------

header =st.container()
model_training  =st.container()

with header:
    st.header('Diet')
    st.text('start with me')
    
    # to preview dataset 
    data = pd.read_csv('marium.csv', index_col=False)
    st.write(data.head())

    st.bar_chart(data.Fat_mass)
    st.line_chart(data.weight)


with model_training:
    st.header('let us predict')
    st.text('Resulted smm')
    

    X =data[['weight']]   #[[]] to make one dimension array 
    y =data[['smm']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lrgr = LinearRegression()
    lrgr.fit(X, y)
    st.write(y)

    

  











