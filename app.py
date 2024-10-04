import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler

# 加载模型和标准化器
model = joblib.load('iris_model.pkl')
scaler = joblib.load('scaler.pkl')

def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)
    return prediction[0]

# 创建 Streamlit 界面
st.title('Iris Flower Prediction')


# 输入字段
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0)

# 预测按钮
if st.button('Predict'):
    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    st.write('Predicted Class:', result)