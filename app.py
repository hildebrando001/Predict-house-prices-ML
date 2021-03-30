import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor


@st.cache
def get_data():
    return pd.read_csv("data.csv")


def train_model():
    data = get_data()
    x = data.drop("MEDV", axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor

# create dataframe
data = get_data()


model = train_model()

# title
st.title("Data App - Predict House Price")
st.write("---")

# subtitle
st.markdown("This Data App uses a machine learning algorithm to predict the house price in Boston.")


st.subheader("Select a set of attributes")

# standard data to be shown
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# Defining attributes from multiselect
cols = st.multiselect("Attributes", data.columns.tolist(), default=defaultcols)

#Show to 10 dataframe data
st.dataframe(data[cols].head(10))


st.subheader("House distribution by their prices")

# Defining range of prices
price_range = st.slider("Price range", float(data.MEDV.min()), float(data.MEDV.max()), (16.0, 30.0))

# Filtering data
fdata = data[data['MEDV'].between(left=price_range[0], right=price_range[1])]

# Plot data distribution
f = px.histogram(fdata, x="MEDV", nbins=100, title="Price Distribution")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total")
st.plotly_chart(f)


st.sidebar.subheader("Fill the attributes")

# mapping Attributes
crim = st.sidebar.number_input("Criminality rate", value=data.CRIM.mean())
indus = st.sidebar.number_input("Proportional size in hectares", value=data.INDUS.mean())
chas = st.sidebar.selectbox("Limits to the rive?", ("Yes","No"))
# changing entries data to a binary value
chas = 1 if chas == "Yes" else 0
nox = st.sidebar.number_input("Concentration of nitric oxide", value=data.NOX.mean())
rm = st.sidebar.number_input("Number of rooms", value=1)
ptratio = st.sidebar.number_input("Number of students per teacher", value=data.PTRATIO.mean())
b = st.sidebar.number_input("Percent of Afro American people", value=data.B.mean())
lstat = st.sidebar.number_input("Low status percent", value=data.LSTAT.mean())
medv = data.MEDV.mean()


# Insert button
btn_predict = st.sidebar.button("Run Algotithm")

# verify if button is clicked
if btn_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat,medv]])
    st.subheader("The predict price to the house is:")
    result = "US $ "+str(round(result[0]*10,2))
    st.write(result)
