import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import base64
import io
from io import BytesIO

st.title('Applicability domain for 2D-QSAR models based on leverages')

st.markdown("""
            This web application plots a williams plot for your QSAR model
            * **Python libraries :** Holy shit I'm Adnane --' """ )

st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
    Applicability domain prediction for QSAR models
    """)
st.sidebar.header('User Input Parameters')

def user_input_features():
    b = st.sidebar.slider('Coefficient of AD', 2.0,3.0,2.5)
    data = {'Coefficient of AD': b}
    features = float(b)
    return features
b = user_input_features()

def hat_matrix(X1):#, X2): #Hat Matrix
    hat_mat =  np.dot(np.dot(X1, np.linalg.inv(np.dot(X1.T, X1))), X1.T)
    return hat_mat



def williams_plot(X_train, X_test, Y_true_train, Y_true_test, y_pred_train, y_pred_test, toPrint =True,toPlot=True):
    # H_train= hat_matrix(np.concatenate([X_train, X_test], axis=0))#, numpy.concatenate([X_train, X_test], axis=0))
    X_train=pd.DataFrame(X_train)
    X_test=pd.DataFrame(X_test)
    H_train2= hat_matrix(np.concatenate([X_train, X_test], axis=0))
    H_train= hat_matrix (X_train)
    H_test= hat_matrix (X_test)

    y_pred_test=pd.DataFrame(y_pred_test).to_numpy()
    y_pred_train=pd.DataFrame(y_pred_train).to_numpy()
    Y_true_train=pd.DataFrame(Y_true_train).to_numpy()
    Y_true_test=pd.DataFrame(Y_true_test).to_numpy()


    y_pred_test = y_pred_test.reshape(y_pred_test.shape[0],)
    y_pred_train = y_pred_train.reshape(y_pred_train.shape[0],)
    Y_true_train = Y_true_train.reshape(Y_true_train.shape[0],)
    Y_true_test = Y_true_test.reshape(Y_true_test.shape[0],)

    residual_train= np.abs(Y_true_train - y_pred_train)
    residual_test= np.abs(Y_true_test - y_pred_test)
    s_residual_train = ((residual_train) - np.mean(residual_train)) / np.std(residual_train)
    s_residual_test = (residual_test - np.mean(residual_test))/ np.std(residual_test)

    leverage= np.diag(H_train)
    leverage_train = leverage[0:X_train.shape[0]]
    leverage_test = leverage[0:X_test.shape[0]]
    p = X_train.shape[1] #features
    n = X_train.shape[0] #+ X_test.shape[0] #training compounds
    try :
        h_star = (float(b) * (p+1))/float(n)
    except ZeroDivisionError :
        pass

    train_points_in_ad = float(100 * np.sum(np.asarray(leverage_train < h_star) & np.asarray(s_residual_train<3))) / len(leverage_train)
    test_points_in_ad = float(100 * np.sum(np.asarray(leverage_test < h_star) & np.asarray(s_residual_test<3))) / len(leverage_test)

    test_lev_out = np.sum(np.asarray(leverage_test > h_star))

    if toPrint:
      st.write("Percetege of train points inside AD: {}%".format(train_points_in_ad))
      st.write("Percetege of test points inside AD: {}%".format(test_points_in_ad))
      st.write("h*: {}".format(h_star))


    if toPlot:

      plt.plot(leverage_train.tolist(),s_residual_train.tolist(),'o', label='train')
      plt.plot(leverage_test.tolist(),s_residual_test.tolist(),'^', label = 'test')
      plt.axhline(y=3, color='r', linestyle='-')
      plt.axhline(y=-3, color='r', linestyle='-')
      plt.axvline(x=h_star, color='k', linestyle='--')
      #plt.ylim(bottom=-6)
      plt.title('williams plot')
      plt.xlabel('Leverage')
      plt.ylabel('Standardized Residuals')
      plt.legend(loc='lower right', shadow=True)
      fig = plt.figure('williams_plot_9edour.pdf')
      plt.close()
      #plt.show()
      st.pyplot(plt)
      st.markdown(imagedownload (plt,'williams_plot.pdf'), unsafe_allow_html=True)


      return test_points_in_ad,train_points_in_ad,test_lev_out,h_star,leverage_train,leverage_test,s_residual_train,s_residual_test
def imagedownload (df, filename):
    s = io.BytesIO()
    plt.savefig(s, format = 'pdf', bbox_inches = 'tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

uploaded_file1 = st.sidebar.file_uploader("""Upload your X train""")
if uploaded_file1 is not None:
  uploaded_file1 = pd.read_csv(uploaded_file1)
  #st.write(uploaded_file1.shape)

#n = uploaded_file1.shape[1]

uploaded_file2 = st.sidebar.file_uploader("""Upload your X test""")
if uploaded_file2 is not None:
  uploaded_file2 = pd.read_csv(uploaded_file2)
  #st.write(uploaded_file2.shape)

uploaded_file3 = st.sidebar.file_uploader("""Upload your Y true train""")
if uploaded_file3 is not None:
    uploaded_file3 = pd.read_csv(uploaded_file3)
    #st.write(uploaded_file3.shape)

uploaded_file4 = st.sidebar.file_uploader("""Upload your Y true test""")
if uploaded_file4 is not None:
     uploaded_file4 = pd.read_csv(uploaded_file4)
     #st.write(uploaded_file4.shape)

uploaded_file5 = st.sidebar.file_uploader("""Upload your Y pred train""")
if uploaded_file5 is not None:
    uploaded_file5 = pd.read_csv(uploaded_file5)
    #st.write(uploaded_file5.shape)

uploaded_file6 = st.sidebar.file_uploader("""Upload your Y pred test""")
if uploaded_file6 is not None:
    uploaded_file6 = pd.read_csv(uploaded_file6)
    ##st.write(y_pred_test)

# if st.button('Calculate AD '):
williams_plot(X_train = uploaded_file1 , X_test = uploaded_file2, Y_true_train = uploaded_file3, Y_true_test = uploaded_file4, y_pred_train = uploaded_file5,y_pred_test = uploaded_file6, toPrint =True,toPlot=True)
   #st.write(results)
