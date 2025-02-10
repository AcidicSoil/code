import ollama

def generate_code(prompt):
    model = "qwen2.5-coder:1.5b"  # Change to `mistral` or other if needed
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# Example Usage
print(generate_code("Write a Python function to reverse a string."))
