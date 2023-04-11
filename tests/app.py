# This script is though to demo gradio
import gradio as gr

# help(gr)

def greet(name):
    return 'Hello ' + name + ' !'

demo = gr.Interface(
    fn=greet,
    inputs='text',
    outputs='text'
)

demo.launch()