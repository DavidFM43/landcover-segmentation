import gradio as gr

# Build segmentation model

def segment(img):
    print(type(img))
    return img

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
