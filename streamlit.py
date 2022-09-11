import streamlit as st
from st_functions import *
from streamlit_option_menu import option_menu
from keras.models import load_model


with st.sidebar:
    selection_screen = option_menu("Liver Deliver", ["Home", "About", "Predict"], icons=['house', 'info', 'gear'],menu_icon='lens', default_index=1)
if selection_screen == "Home":
    st.title("  Liver Deliver")
    st.write(" ---")
    st.write("#### Diese Website beinhaltet mein Projekt für den Bundeswettbewerb Künstliche Intelligenz 2022. \
    Hier können Sie die Segmentation mit 3 verschiedenen Modellen testen.")
    st.write(" ---")
    st.write("### U-Net")
    st.write("#### Das U-Net ist ein Encoder-Decoder-Modell, das für die Segmentierung von Bildern verwendet wird. \
        Es besteht aus einem Encoder, der die Bildinformationen in einem Featurevektor zusammenfasst, und einem Decoder, \
            der die Informationen wieder in ein Bild umwandelt. Das U-Net ist ein sehr beliebtes Modell für die Segmentierung, \
                da es sehr gut funktioniert und relativ einfach zu implementieren ist.\
                    Das U-Net was Sie hier aber antreffen ist tatsächlich das Modell mit den meisten Parametern, \
                        da es Filter in der Reichweite von 64 bis 1024 und somit auch eine Anzahl von über 9 Millionen Parametern hat. ")
    st.write(" ---")
    st.write("### Residual Attention U-Net")
    st.write("#### Das Residual Attention U-Net basiert auf dem U-Net, jedoch mit Residual-Verbindungen und Attention-Modulen.\
        Residual-Verbindungen sind eine Art Skip-Connection, die es dem Modell ermöglichen, die Informationen aus den verschiedenen \
            Schichten zu erhalten. Die Attention-Module sind eine Art von Feature-Map-Verbindung, die es dem Modell ermöglichen, \
                sich auf bestimmte Bereiche des Bildes zu konzentrieren.  \
                    Die letzten beiden Modelle sind von dieser Art, beinhalten jedoch verschiedene Layer \
                        weshalb sie unterschiedliche Ergebnisse liefern. \
                         ResAttUnet1 hat rund 5.4 Millionen Parameternund schließt mit einem Global Average Pooling Layer und Dense Layern ab.\
                                ResAttUnet2 hat rund 2.5 Millionen Parameter und ist somit das kleinste Netzwerk mit den wenigsten Filtern.\
                            ")
    st.write(" ---")
    st.write("### Klassifizierung der Tumore")
    st.write("#### Die Klassifizierung der Segmentationen ist leider sehr oberflächlich und nicht sehr genau. \
        Sie beruht lediglich auf Daten, die von den Pixeln der Segmentationen abgeleitet werden. \
            U-Net ist hierbei jedoch das Model, das am besten abschneidet, da es mit binären Masken trainiert wurde und die Pixel somit\
                besser nutzbar sind. ")


if selection_screen == "Predict":
    st.sidebar.title("Einstellungen")

    model_name = st.sidebar.selectbox("Modelauswahl", ["U-Net", "AttResUnet1", "AttResUnet2"])
    
    upload_settings = st.selectbox("Upload Settings", ["Test Bilder", "Eigenes Bild"])
    if upload_settings == "Test Bilder":
        test_img = st.radio("Test Bilder", ["Bild 1", "Bild 2", "Bild 3"])
        if test_img == "Bild 1":
            img = Image.open("volume-0_slice_58.jpg")
            st.image(img, width=400, caption="Bild 1")
        elif test_img == "Bild 2":
            img = Image.open("volume-0_slice_1.jpg")
            st.image(img, width=400, caption="Bild 2")
        elif test_img == "Bild 3":
            img = Image.open("volume-100_slice_361.jpg")
            st.image(img, width=400, caption="Bild 3")


    elif upload_settings == "Eigenes Bild":
        file_name = st.file_uploader("Bild hochladen", type=["png", "jpg", "jpeg"])
        if file_name is not None:
            img = Image.open(file_name)
            img = img.resize((128,128))
            st.image(img, caption="Uploaded Image", width=400)

    button = st.sidebar.button("Segmentieren")



    def load_model_(model):
        return load_model(f"/home/lorenz/U-Net/U-Net-Keras/checkpoints/{model}_model.h5", custom_objects={ "dice_score": dice_score , "K": K, "recall_m": recall_m, "precision_m": precision_m, "f1_m": f1_m})

    def return_color(mask):
        mask = mask[0, :, :, :] 
        mask1 = colorize_mask(mask)
        st.image(mask1, caption="Colored", width=400)

    def classify(mask):
        st.sidebar.write(f"Größe des Tumors in Pixeln: {tumor_size(mask)}")
        st.sidebar.write(f"Prozentzahl, die der Tumor von der ganzen Maske bedeckt: {tumor_size_percentage(mask)*100}%")
        st.sidebar.write(f"Unegfähre Prozentzahl, die der Tumor von der Leber bededckt: {tumor_surface_percentage(mask)*100}%")

    if model_name == "U-Net":
        model = load_model_("Unet")
    elif model_name == "AttResUnet1":
        model = load_model_("ResAttUnet1")
    elif model_name == "AttResUnet2":
        model = load_model_("final_ARU2")

    if button:
        if model_name == "U-Net":
            img = np.array(img)
            mask = predict(img, model=model)
            st.image(mask, caption="Mask", width=400)
            classify(mask)
            return_color(mask)
        if model_name == "AttResUnet2":
            img = np.array(img)
            mask = att_res_pred(img, model=model)
            st.image(mask, caption="Mask", width=400)
            classify(mask)
            return_color(mask)
        if model_name == "AttResUnet1":
            img = np.array(img)
            mask = att_pred(img, model=model)
            st.image(mask, caption="Mask", width=400)
            classify(mask)
            return_color(mask)

if selection_screen == "About":
    st.write("#### This webside and it's content has been made with the help of multiple sources which can be found *inside the readme\
    in the corresponding github repository.* The side is made with Streamlit and the models are made with Keras and Tensorflow.\
        ")
    st.write("#### Usage under own risk. This side is not meant to be used for medical purposes.\
    ")
    st.write("#### Made by Lorenz Rutkevich [Github](https://github.com/lorenz-7)")








