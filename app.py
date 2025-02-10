import gradio as gr
import subprocess
from typing import List, Dict
import ollama
import json
import time


def get_models() -> List[Dict[str, str]]:
    """
    Fetch available Ollama models using direct command line interface
    Returns list of dicts with model information
    """
    try:
        # Use subprocess to run 'ollama list' command
        result = subprocess.run(
            ['ollama', 'list'], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Error running ollama list: {result.stderr}")
            return [{'name': 'llama2', 'full_name': 'llama2:7b', 'size': 'Unknown', 'modified': 'Unknown'}]

        # Parse the output
        models = []
        lines = result.stdout.strip().split('\n')

        # Skip the header line if it exists
        start_idx = 1 if 'NAME' in lines[0] else 0

        for line in lines[start_idx:]:
            if not line.strip():
                continue

            # Split the line and handle potential spaces in names
            parts = line.split()
            if len(parts) >= 4:  # Ensure we have at least name, size, and modified date
                name = parts[0]
                size = parts[1] + ' ' + parts[2]
                # Join the remaining parts as the modified date
                modified = ' '.join(parts[3:])

                models.append({
                    'name': name.split(':')[0],  # Base name without tag
                    'full_name': name,
                    'size': size,
                    'modified': modified
                })

        return models if models else [{'name': 'llama2', 'full_name': 'llama2:7b', 'size': 'Unknown', 'modified': 'Unknown'}]

    except Exception as e:
        print(f"Error fetching models: {str(e)}")
        return [{'name': 'llama2', 'full_name': 'llama2:7b', 'size': 'Unknown', 'modified': 'Unknown'}]


def chat_with_model(message: str, history: List[List[str]], model_name: str, temperature: float) -> str:
    """
    Send a message to the selected Ollama model and get a response
    """
    try:
        system_message = """You are an AI programming assistant. When asked about code:
        1. Provide clear, well-commented code examples
        2. Explain your reasoning and approach
        3. Highlight best practices and potential pitfalls
        4. Use markdown formatting for code blocks
        """

        # Format the conversation history
        messages = [
            {"role": "system", "content": system_message}
        ]

        # Add conversation history
        for human_msg, assistant_msg in history:
            messages.append({"role": "user", "content": human_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        # Add the current message
        messages.append({"role": "user", "content": message})

        # Call Ollama with the correct parameters
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={
                "temperature": temperature
            }
        )

        return response['message']['content']
    except Exception as e:
        print(f"Debug - Error in chat_with_model: {str(e)}")
        return f"Error: {str(e)}"


def create_interface():
    initial_models = get_models()

    with gr.Blocks(theme=gr.themes.Base(primary_hue="blue", secondary_hue="neutral")) as interface:
        gr.Markdown("# AI-Assisted Code Generator")

        # Model selection section
        with gr.Row():
            with gr.Column(scale=2):
                model_info = gr.Markdown("### Available Models")
                model_table = gr.Dataframe(
                    headers=["Model", "Full Name", "Size", "Last Modified"],
                    value=[[m['name'], m['full_name'], m['size'], m['modified']]
                           for m in initial_models],
                    label="Available Models"
                )

                model_dropdown = gr.Dropdown(
                    choices=[m['full_name'] for m in initial_models],
                    value=initial_models[0]['full_name'] if initial_models else None,
                    label="Select Model",
                    interactive=True
                )

                temperature = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.7,
                    label="Temperature"
                )

                refresh_button = gr.Button("🔄 Refresh Models")

        # Chat interface section
        gr.Markdown("### Chat Interface")
        chatbot = gr.Chatbot(
            label="Chat History",
            height=500,
            show_copy_button=True,
            bubble_full_width=False
        )

        with gr.Row():
            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask about code, programming concepts, or request code examples...",
                lines=3
            )
            submit = gr.Button("Send", variant="primary")

        clear = gr.Button("Clear Chat")

        # Event handlers
        def respond(message, chat_history, model_name, temp):
            if not message:
                return "", chat_history

            bot_message = chat_with_model(
                message, chat_history, model_name, temp)
            chat_history.append((message, bot_message))
            return "", chat_history

        submit.click(
            respond,
            [msg, chatbot, model_dropdown, temperature],
            [msg, chatbot]
        )

        msg.submit(
            respond,
            [msg, chatbot, model_dropdown, temperature],
            [msg, chatbot]
        )

        clear.click(lambda: None, None, chatbot, queue=False)

        def refresh_models():
            new_models = get_models()
            return [
                [[m['name'], m['full_name'], m['size'], m['modified']]
                    for m in new_models],
                gr.Dropdown(choices=[m['full_name'] for m in new_models])
            ]

        refresh_button.click(
            refresh_models,
            outputs=[model_table, model_dropdown]
        )

        return interface


if __name__ == "__main__":
    # Ensure Ollama is running
    print("Starting AI-Assisted Code Generator...")
    print("Checking Ollama service...")

    interface = create_interface()
    interface.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=True
    )
