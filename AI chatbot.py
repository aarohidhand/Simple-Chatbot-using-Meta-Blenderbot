import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

conversation_history = []

def chat_with_bot(input_text):
    global conversation_history
    
    history_string = "\n".join(conversation_history)
    
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    
    outputs = model.generate(**inputs)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    conversation_history.append(f"User: {input_text}")
    conversation_history.append(f"Bot: {response}")
    
    return "\n".join(conversation_history)

with gr.Blocks() as demo:
    gr.Markdown("<h1><center>Text with BlenderBot</center></h1>")
    
    chatbot = gr.Textbox(label="Conversation", value="", placeholder="Conversation will appear here...", lines=20, interactive=False)
    user_input = gr.Textbox(label="Your message")
    send_button = gr.Button("Send")

    def update_conversation(input_text):
        return chat_with_bot(input_text)
    
    send_button.click(update_conversation, inputs=user_input, outputs=chatbot)
    user_input.submit(update_conversation, inputs=user_input, outputs=chatbot)
    
demo.launch()
