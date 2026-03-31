import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# 1. 网页标题和描述
st.title("🐱 猫狗大战：AI 图像识别器 🐶")
st.write("上传一张图片，让 AI 告诉你它是猫还是狗！")

# 2. 加载模型（使用缓存避免重复加载）
@st.cache_resource
def load_my_model():
    # 确保文件名和你保存的一致
    model = tf.keras.models.load_model('model1_catsVSdogs_10epoch.h5')
    return model

model = load_my_model()

# 3. 侧边栏或主界面上传图片
uploaded_file = st.file_uploader("请选择一张图片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 展示用户上传的图片
    image = Image.open(uploaded_file)
    st.image(image, caption='上传的图片', use_container_width=True)
    st.write("正在识别中...")

    # 4. 图片预处理（必须和训练时完全一致）
    # 假设你训练时用的是 128x128
    img = image.resize((128, 128)) 
    img_array = np.array(img)
    img_array = img_array / 255.0  # 归一化
    img_array = np.expand_dims(img_array, axis=0) # 增加批次维度

    # 5. 进行预测
    prediction = model.predict(img_array)
    result_index = np.argmax(prediction, axis=-1)[0]
    
    # 6. 显示结果
    results = {0: '猫 (Cat) 🐱', 1: '狗 (Dog) 🐶'}
    confidence = np.max(prediction) * 100 # 置信度（概率）

    st.success(f"识别结果：该图片有 {confidence:.2f}% 的概率是 **{results[result_index]}**")