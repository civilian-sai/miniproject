import streamlit as st
import pickle
import numpy as np
model=pickle.load(open('model1.pkl','rb'))



def predict_defect(x1,x2,x3,x4,x5,x6,x7):
    input=np.array([[x1,x2,x3,x4,x5,x6,x7]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    print(type(pred))
    return float(pred)

def main():
    st.title("Streamlit tutorial")
    html_temp= """
              <div style="background-color:#025246; padding:10px">
              <h2 style="color:white;text-align:center;">Software Defect Detection</h2>
              </div>
               """
    st.markdown(html_temp, unsafe_allow_html=True)
    x1=st.text_input("x1","type here")
    x2=st.text_input("x2","type here")
    x3=st.text_input("x3","type here")
    x4=st.text_input("x4","type here")
    x5=st.text_input("x5","type here")
    x6=st.text_input("x6","type here")
    x7=st.text_input("x7","type here")
    safe_html="""
              <div style="background-color:#F4D03f;padding:10px>
              <h2 sstyle="color:white;text-align:center;">code is safe</h2>
              </div>
              """
    danger_html="""
     <div style="background-color:#F08080;padding:10px>
     <h2 sstyle="color:black;text-align:center;">code is defected</h2>
     </div>
     """
    if st.button("predict"):
        output=predict_defect(x1,x2,x3,x4,x5,x6,x7)
        if output>0.5:
            st.markdown(danger_html,unsafe_allow_html=True)
        else:
            st.markdown(safe_html,unsafe_allow_html=True)

if __name__=='__main__':
    main()