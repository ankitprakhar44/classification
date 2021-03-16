import streamlit as st
import tensorflow as tf
from tensorflow import keras

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)


def load_model1():
    model1=tf.keras.models.load_model(tf.keras.utils.get_file('orchid_species_model.hdf5', origin='https://drive.google.com/u/1/uc?id=1Ub-EESyKhqxod8SVgnTo5q6oggeyKhEm&export=download'))
    return model1
model1=load_model1()

def load_model2():
    model2=tf.keras.models.load_model(tf.keras.utils.get_file('cymbidium_model.hdf5', origin='https://drive.google.com/u/1/uc?id=1QO8cClJURkUmrHzREvlx0ahnhUMw4n7Q&export=download'))
    return model2
model2=load_model2()


st.write("""
         # Orchid Species Classification
         """
        )


file = st.file_uploader("Please upload a image of Orchid", type=["jpg","png","jpeg"])
from PIL import Image, ImageOps
import numpy as np


def import_and_predict1(image_data, model1):

    size = (180,180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction1 = model1.predict(img_reshape)

    return prediction1

def import_and_predict2(image_data, model2):

    size = (180,180)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction2 = model2.predict(img_reshape)

    return prediction2


https://drive.google.com/u/0/uc?id=1GkVfMC4qrfgjvqVYxIJyzPwMS6_jlYz3&export=download
['Cymbidium acuminatum', 'Cymbidium aliciae', 'Cymbidium aloifolium', 'Cymbidium atropurpureum', 'Cymbidium banaense', 'Cymbidium bicolor', 'Cymbidium borneense', 'Cymbidium canaliculatum', 'Cymbidium changningense', 'Cymbidium chloranthum', 'Cymbidium cochleare', 'Cymbidium concinnum', 'Cymbidium crassifolium', 'Cymbidium cyperifolium', 'Cymbidium daweishanense', 'Cymbidium dayanum', 'Cymbidium defoliatum', 'Cymbidium devonianum', 'Cymbidium eburneum', 'Cymbidium elegans', 'Cymbidium elongatum', 'Cymbidium ensifolium', 'Cymbidium erythraeum', 'Cymbidium erythrostylum', 'Cymbidium faberi', 'Cymbidium finlaysonianum', 'Cymbidium floribundum', 'Cymbidium formosanum', 'Cymbidium gaoligongense', 'Cymbidium goeringii', 'Cymbidium haematodes', 'Cymbidium hartinahianum', 'Cymbidium hookerianum', 'Cymbidium insigne', 'Cymbidium iridioides', 'Cymbidium kanran', 'Cymbidium lancifolium', 'Cymbidium lowianum', 'Cymbidium macrorhizon', 'Cymbidium madidum', 'Cymbidium maguanense', 'Cymbidium mastersii', 'Cymbidium micranthum', 'Cymbidium munronianum', 'Cymbidium nanulum', 'Cymbidium omeiense', 'Cymbidium parishii', 'Cymbidium qiubeiense', 'Cymbidium rectum', 'Cymbidium repens', 'Cymbidium sanderae', 'Cymbidium schroederi', 'Cymbidium seidenfadenii', 'Cymbidium serratum', 'Cymbidium sichuanicum', 'Cymbidium sigmoideum', 'Cymbidium sinense', 'Cymbidium suave', 'Cymbidium suavissimum', 'Cymbidium tamphianum', 'Cymbidium tigrinum', 'Cymbidium tortisepalum', 'Cymbidium tracyanum', 'Cymbidium wadae', 'Cymbidium wenshanense', 'Cymbidium whiteae', 'Cymbidium wilsonii', 'Cymbidium × ballianum', 'Cymbidium × baoshanense', 'Cymbidium × dilatatiphyllum', 'Cymbidium × florinda', 'Cymbidium × gammieanum', 'Cymbidium × glebelandense', 'Cymbidium × hillii', 'Cymbidium × monanthum', 'Cymbidium × nishiuchianum', 'Cymbidium × nomachianum', 'Cymbidium × purpuratum', 'Cymbidium × rosefieldense', 'Cymbidium × woodlandense']
    
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions1 = import_and_predict1(image, model1)
    predictions2 = import_and_predict2(image, model2)
    class_names1=['anoectochilus  burmanicus rolfe', 'bulbophyllum auratum Lindl', 'bulbophyllum auricomum  lindl', 'bulbophyllum dayanum rchb', 'bulbophyllum lasiochilum par. & rchb', 'bulbophyllum limbatum', 'bulbophyllum longissimum (ridl.) ridl', 'bulbophyllum medusae (lindl.) rchb', 'bulbophyllum morphologorum F.Kranzl', 'bulbophyllum patens  king ex hk.f', 'bulbophyllum rufuslabram', 'bulbophyllum siamensis rchb', 'calenthe rubens', 'chiloschista parishii seidenf', 'chiloschista viridiflora seidenf', 'cymbidium', 'dendrobium chrysotoxum lindl', 'dendrobium cumulatum Lindl', 'dendrobium farmeri paxt', 'dendrobium fimbriatum  hook', 'dendrobium lindleyi steud', 'dendrobium pulchellum', 'dendrobium pulchellum roxb', 'dendrobium secundum bl-lindl', 'dendrobium senile par. & rchb.f', 'dendrobium signatum rchb. f', 'dendrobium thyrsiflorum rchb. f', 'dendrobium tortile', 'dendrobium tortile lindl', 'hygrochillus parishii var. marrioftiana (rchb.f.)', 'maxiralia tenui folia', 'oncidium goldiana', 'paphiopedilum bellatulum', 'paphiopedilum callosum', 'paphiopedilum charlesworthii', 'paphiopedilum concolor', 'paphiopedilum exul', 'paphiopedilum godefroyae', 'paphiopedilum gratrixianum', 'paphiopedilum henryanum', 'paphiopedilum intanon-villosum', 'paphiopedilum niveumя(rchb.f.) stein', 'paphiopedilum parishii', 'paphiopedilum spicerianum', 'paphiopedilum sukhakulii', 'paphiopedilum vejvarutianum O. Gruss & Roellke', 'pelatantheria bicuspidata  (rolfe ex downie) tang & wang', 'pelatantheria insectiflora (rchb.f.) ridl', "phaius tankervilleaeя(banks ex i' heritier) blume", 'phalaenopsis cornucervi (breda) bl. & rchb.f', 'rhynchostylis gigantea (lindl.) ridl', 'trichoglottis orchideae (koern) garay']
    class_names2=['Cymbidium acuminatum', 'Cymbidium aliciae', 'Cymbidium aloifolium', 'Cymbidium atropurpureum', 'Cymbidium banaense', 'Cymbidium bicolor', 'Cymbidium borneense', 'Cymbidium canaliculatum', 'Cymbidium changningense', 'Cymbidium chloranthum', 'Cymbidium cochleare', 'Cymbidium concinnum', 'Cymbidium crassifolium', 'Cymbidium cyperifolium', 'Cymbidium daweishanense', 'Cymbidium dayanum', 'Cymbidium defoliatum', 'Cymbidium devonianum', 'Cymbidium eburneum', 'Cymbidium elegans', 'Cymbidium elongatum', 'Cymbidium ensifolium', 'Cymbidium erythraeum', 'Cymbidium erythrostylum', 'Cymbidium faberi', 'Cymbidium finlaysonianum', 'Cymbidium floribundum', 'Cymbidium formosanum', 'Cymbidium gaoligongense', 'Cymbidium goeringii', 'Cymbidium haematodes', 'Cymbidium hartinahianum', 'Cymbidium hookerianum', 'Cymbidium insigne', 'Cymbidium iridioides', 'Cymbidium kanran', 'Cymbidium lancifolium', 'Cymbidium lowianum', 'Cymbidium macrorhizon', 'Cymbidium madidum', 'Cymbidium maguanense', 'Cymbidium mastersii', 'Cymbidium micranthum', 'Cymbidium munronianum', 'Cymbidium nanulum', 'Cymbidium omeiense', 'Cymbidium parishii', 'Cymbidium qiubeiense', 'Cymbidium rectum', 'Cymbidium repens', 'Cymbidium sanderae', 'Cymbidium schroederi', 'Cymbidium seidenfadenii', 'Cymbidium serratum', 'Cymbidium sichuanicum', 'Cymbidium sigmoideum', 'Cymbidium sinense', 'Cymbidium suave', 'Cymbidium suavissimum', 'Cymbidium tamphianum', 'Cymbidium tigrinum', 'Cymbidium tortisepalum', 'Cymbidium tracyanum', 'Cymbidium wadae', 'Cymbidium wenshanense', 'Cymbidium whiteae', 'Cymbidium wilsonii', 'Cymbidium × ballianum', 'Cymbidium × baoshanense', 'Cymbidium × dilatatiphyllum', 'Cymbidium × florinda', 'Cymbidium × gammieanum', 'Cymbidium × glebelandense', 'Cymbidium × hillii', 'Cymbidium × monanthum', 'Cymbidium × nishiuchianum', 'Cymbidium × nomachianum', 'Cymbidium × purpuratum', 'Cymbidium × rosefieldense', 'Cymbidium × woodlandense']
    
    if class_names1[np.argmax(predictions1)] == 'cymbidium':
        string="This image most likely belongs to cymbidium gene and most likely comes under the category of: "+class_names2[np.argmax(predictions2)]
        st.success(string)
    else:
        string="This image most likely belongs to the specie of: "+class_names1[np.argmax(predictions1)]
        st.success(string)
