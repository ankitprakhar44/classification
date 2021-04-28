import streamlit as st
import tensorflow as tf


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(tf.keras.utils.get_file('new_cymbidium_model_1.hdf5', origin='https://drive.google.com/u/0/uc?id=1GkVfMC4qrfgjvqVYxIJyzPwMS6_jlYz3&export=download'))
    return model
model=load_model()
st.write("""
         # Cymbidium Orchid Classification
         """
        )
file = st.file_uploader("Please upload an image of Cymbidium Orchid", type=["jpg","png","jpeg"])
from PIL import Image, ImageOps
import numpy as np
def import_and_predict(image_data, model):

    size = (180,180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)

    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names=['Cymbidium acuminatum', 'Cymbidium aliciae', 'Cymbidium aloifolium', 'Cymbidium atropurpureum', 'Cymbidium banaense', 'Cymbidium bicolor', 'Cymbidium borneense', 'Cymbidium canaliculatum', 'Cymbidium changningense', 'Cymbidium chloranthum', 'Cymbidium cochleare', 'Cymbidium concinnum', 'Cymbidium crassifolium', 'Cymbidium cyperifolium', 'Cymbidium daweishanense', 'Cymbidium dayanum', 'Cymbidium defoliatum', 'Cymbidium devonianum', 'Cymbidium eburneum', 'Cymbidium elegans', 'Cymbidium elongatum', 'Cymbidium ensifolium', 'Cymbidium erythraeum', 'Cymbidium erythrostylum', 'Cymbidium faberi', 'Cymbidium finlaysonianum', 'Cymbidium floribundum', 'Cymbidium formosanum', 'Cymbidium gaoligongense', 'Cymbidium goeringii', 'Cymbidium haematodes', 'Cymbidium hartinahianum', 'Cymbidium hookerianum', 'Cymbidium insigne', 'Cymbidium iridioides', 'Cymbidium kanran', 'Cymbidium lancifolium', 'Cymbidium lowianum', 'Cymbidium macrorhizon', 'Cymbidium madidum', 'Cymbidium maguanense', 'Cymbidium mastersii', 'Cymbidium micranthum', 'Cymbidium munronianum', 'Cymbidium nanulum', 'Cymbidium omeiense', 'Cymbidium parishii', 'Cymbidium qiubeiense', 'Cymbidium rectum', 'Cymbidium repens', 'Cymbidium sanderae', 'Cymbidium schroederi', 'Cymbidium seidenfadenii', 'Cymbidium serratum', 'Cymbidium sichuanicum', 'Cymbidium sigmoideum', 'Cymbidium sinense', 'Cymbidium suave', 'Cymbidium suavissimum', 'Cymbidium tamphianum', 'Cymbidium tigrinum', 'Cymbidium tortisepalum', 'Cymbidium tracyanum', 'Cymbidium wadae', 'Cymbidium wenshanense', 'Cymbidium whiteae', 'Cymbidium wilsonii', 'Cymbidium × ballianum', 'Cymbidium × baoshanense', 'Cymbidium × dilatatiphyllum', 'Cymbidium × florinda', 'Cymbidium × gammieanum', 'Cymbidium × glebelandense', 'Cymbidium × hillii', 'Cymbidium × monanthum', 'Cymbidium × nishiuchianum', 'Cymbidium × nomachianum', 'Cymbidium × purpuratum', 'Cymbidium × rosefieldense', 'Cymbidium × woodlandense']
    string_percent= np.argmax(predictions)
    string="It is "+"% likely that his image belongs to the specie: "+class_names[np.argmax(predictions)]
    st.success(string)
    
    
