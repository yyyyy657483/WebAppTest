import gradio as gr
from gradioWebApp import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)