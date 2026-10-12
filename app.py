
# Student Roll No: 24BAD048
# AI Chatbot using Qwen2.5, PyTorch and Gradio
from datetime import datetime
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Select available device
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Student Roll No: 24BAD048")
print("Loading model on:", device)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

print("Model loaded successfully!")



def chatbot_response(message, history):
    if not message or not message.strip():
        return "Please enter a message."

    # Handle today's date directly
    if "date" in message.lower() and "today" in message.lower():
        return "Today's date is " + datetime.now().strftime("%d %B %Y")

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        }
    ]

    # Keep your existing history-processing code here.

    messages.append({
        "role": "user",
        "content": message.strip()
    })

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        answer = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        return answer or "No response was generated."

    except Exception as error:
        import traceback
        traceback.print_exc()
        return f"Error: {type(error).__name__}: {error}"



# Create the Gradio chatbot interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="AI Chatbot - 24BAD048",
    description="A conversational AI chatbot powered by Qwen2.5-0.5B-Instruct.",
    examples=[
        "What is artificial intelligence?",
        "Explain Python loops.",
        "Give me three study tips."
    ]
)

# Launch the application
if __name__ == "__main__":
    demo.launch()
