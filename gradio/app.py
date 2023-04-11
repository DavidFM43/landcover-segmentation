import gradio as gr
import numpy as np

# Build segmentation model

def normalize_img(img, new_shape=(512, 512, 3)):
    "Normalize a np array in order to have the same Unet's input"
    return np.resize(img, new_shape=new_shape)

def segment(img):
    img_type = type(img)
    img_shape = img.shape

    return str(img_shape)

def main():
    # Create the app interface
    gradio_interface = gr.Interface(
                                fn=segment,
                                inputs='image',
                                outputs='image'
                            )
    # Launch interface
    gradio_interface.launch(share=True)

if __name__=='__main__':
    print('Running')
    main()
