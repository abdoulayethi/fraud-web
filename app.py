import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import pickle
from catboost import CatBoostClassifier
import shap  # package used to calculate Shap values
import matplotlib.pyplot as plt



st.title('DETECTION DE FRAUDE BICIS')

DATE_COLUMN = 'END_TIME'
DATA_URL = "./sample_data.csv"

@st.cache()
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    #data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

#@st.cache()
def load_model():
    #open('final_model.pkl','rb')
    model = CatBoostClassifier()
    model.load_model("fraud_model")
    #encoder = joblib.load("encoder.pkl")
    return model
   
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

def process_data(data):
    index = data.index
    cols_to_drop = ['end_time','P022','ECI','NUM_CARTE_Alea','NUM_COMPTE_alea','EXT_RES_CODE','NUM_AUTOR','nom_pays','devise',
                    'transaction_date','transaction_time','transaction_datetime','transaction_minute','code_pays','montant_en_dollard',
                    'MONTANT_DOLLARD','NOM_AQUERREUR','TYPE_TXN','taux_conversion','montant_debit_moyen_par_trans_7jrs',
                    'montant_debit_moyen_par_trans_15jrs','montant_debit_moyen_par_jour_7jrs','montant_debit_moyen_par_jour_15jrs',
                    'montant_debit_moyen_par_jour_30jrs','montant_debit_moyen_par_trans_30jrs','true_class','pred_class','fraud_prob',"ville_aquerreur","acquirer"]
    cols_to_drop = [val.lower() for val in cols_to_drop]
    encoded_data = data.drop(cols_to_drop,axis=1)
    
    encoded_data.columns = [val.replace("_encoded","") for val in encoded_data.columns]
    #st.write("data columns", encoded_data.columns)
    return encoded_data
    
def predict(model,data):

    probabilities = model.predict_proba(data)
    corrected_predictions = []
    fraud_probabilities = []
    for prob in probabilities:
        fraud_probabilities.append(round(prob[-1],5)*100)
        if(prob[-1]>=0.70):
            corrected_predictions.append(1)
        else:
            corrected_predictions.append(0)
    return corrected_predictions,fraud_probabilities

st.set_option('deprecation.showPyplotGlobalUse', False)
class_dict = {
    0: "normale",
    1: "suspicieuse"
}
data_load_state = st.text('Chargement du model...')
model = load_model()
# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)


data_load_state.text("Chargement réussi")
data_load_state.text('chargement des données...')
raw_data = load_data(100)
data_load_state.text("Chargement réussi")

if st.checkbox('Afficher des donnnées'):
    st.subheader('Aperçu des données brutes')
    #st.write(data)
    st.dataframe(raw_data)
    option = st.selectbox('Veuillez selectionner une ligne ', raw_data.index)
    st.write('Vous avez sélectionné :', raw_data[option:option+1])
    if st.button('Faire une prediction'):
        encoded = process_data(raw_data[option:option+1])
        prediction,probs = predict(model,encoded)
        # Calculate Shap values
        shap_values = explainer.shap_values(encoded)
        #st.write("prediction du modèke: transaction",class_dict[prediction[0]])
        message = "prediction du modèle: transaction "+class_dict[prediction[0]]
        if prediction[0]==0:
            message+= " avec une probabilité de "+str(100-probs[0])
        else:
            message+= " avec une probabilité de "+str(probs[0])
        st.info(message+"%")
        
        # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
        st.info("Interpétation des prédictions du modèle")
        st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], encoded))
        plt.title('Importannce de chaque variable sur la prédition du modèle')
        shap.summary_plot(shap_values,encoded,plot_type="bar",show=False)
        st.pyplot(bbox_inches='tight')
        plt.clf()
else:
    st.write('Cliquez pour voir les données')
