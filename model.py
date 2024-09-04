import subprocess
import gradio as gr
import requests, json



# model = 'smollm:360m' #You can replace the model name if needed
# context = ["""

#            You are a AI that knows about the cities of india that are given below
            
#                       """] 
context = [] 


def get_installed_models():
    # Run the ollama list command and capture the output
    result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE)
    output = result.stdout.decode("utf-8").splitlines()
    
    # Extract model names from each line, skipping the header
    models = []
    for line in output[1:]:  # Skip the first line (header)
        model_name = line.split()[0]  # Get the first part (model name) of each line
        models.append(model_name)
    
    return models


#Call Ollama API
def generate(prompt, context, top_k, top_p, temp):
    r = requests.post('http://localhost:11434/api/generate',
                     json={
                        "model": "llama2-uncensored",
                        "prompt": prompt,
                        "stream": False,
                        "context": context,
                        "top_k": top_k,
                        "top_p": top_p,
                        "temp": temp
                    },
                     headers={'Content-Type': 'application/json'}
                    )
    r.raise_for_status()

 
    response = ""  

    for line in r.iter_lines():
        body = json.loads(line)
        response_part = body.get('response', '')
        print(response_part)
        if 'error' in body:
            raise Exception(body['error'])

        response += response_part

        if body.get('done', False):
            context = body.get('context', [])
            return response, context



def chat(input, chat_history, top_k, top_p, temp):

    chat_history = chat_history or []

    global context
    output, context = generate(input, context, top_k, top_p, temp)

    chat_history.append((input, output))

    return chat_history, chat_history
  #the first history in return history, history is meant to update the 
  #chatbot widget, and the second history is meant to update the state 
  #(which is used to maintain conversation history across interactions)


#########################Gradio Code##########################
block = gr.Blocks()


with block:

    gr.Markdown("""<h1><center> llama2 </center></h1>
    """)

    
    
    
#     state = gr.State()
    with gr.Row():
        with gr.Column(min_width=600):
#             model = gr.Dropdown(choices=get_installed_models(), label="Select a model")
            chatbot = gr.Chatbot()
            message = gr.Textbox(placeholder="Type here")
            state = gr.State()
            submit = gr.Button("SEND")
        with gr.Column(min_width=100):
            top_k = gr.Slider(0.0,100.0, label="top_k", value=40, info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)")
            top_p = gr.Slider(0.0,1.0, label="top_p", value=0.9, info=" Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)")
            temp = gr.Slider(0.0,2.0, label="temperature", value=0.8, info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)")
    

    

    submit.click(chat, inputs=[message, state, top_k, top_p, temp], outputs=[chatbot, state])
    


block.launch(debug=True,share=True)
