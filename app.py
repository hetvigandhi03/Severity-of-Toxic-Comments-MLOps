
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing import dataTransform

def Predict(text):
    try:
        
        text = dataTransform(text)
        return text
    except Exception as e:
        raise e

def main():
    st.title("Severity of Toxic Comments")
    text = st.text_input("Enter the text")

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            text = Predict(text)
        st.success("Predicted")
        # make text data in percentage
        text[0] = text[0] * 100
        text[0] = text[0].round(2)
        print(text)
        
        df = pd.DataFrame(text, columns = ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate'])

        fig, ax = plt.subplots()
        fig.patch.set_facecolor('#0E1117')

        ax.bar(df.columns, df.iloc[0], color = ['#fe0366', '#3700b3', '#02dbc5', '#f9f343', '#ffb287', '#ba87fc'], 
               width = 0.7, align = 'center', edgecolor = 'white', linewidth = 2, 
               capsize = 10, label = 'Toxicity Percentage', tick_label = df.columns)
        fig.set_size_inches(10, 6)
        ax.set_facecolor('#0E1117')
        ax.set_ylabel('Percentage')
        # change the color of x and y axis
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.tick_params(axis = 'x', colors = 'white')
        ax.tick_params(axis = 'y', colors = 'white')
        # change color of y axis label
        ax.yaxis.label.set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
        
    st.text("Made by: Rounak Bachwani and Hetvi Gandhi")
    # put link
    st.markdown("[Research Paper: Detecting Toxic Comments Using FastText, CNN, and LSTM Models](https://link.springer.com/chapter/10.1007/978-3-031-37940-6_20#Abs1)")
    st.markdown("[GitHub](https://github.com/rounaccc/Severity-of-Toxic-Comments-MLOps)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/rounakb/)")


if __name__ == "__main__":
    main()