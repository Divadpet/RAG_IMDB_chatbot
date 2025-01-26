import tkinter as tk
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
import re

path = "Tekarukite/UHK-IMDB-Dataaset2"

dataset = load_dataset(path=path, split="train")

assert "question" in dataset.column_names and "answer" in dataset.column_names, \
    "Dataset musí mít sloupec 'question' a 'answer'."

questions = dataset["question"]
answers = dataset["answer"]

vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def retrieve_answer(user_question, similarity_threshold=0.3):
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors).flatten()
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]

    if best_match_score >= similarity_threshold:
        return answers[best_match_index], best_match_score
    else:
        return None, best_match_score
    
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
device = model.device

def rag_chatbot(user_question):
    retrieved_answer, similarity_score = retrieve_answer(user_question, similarity_threshold=0.3)

    if retrieved_answer:
        context = retrieved_answer
        print(f"Retrieved answer used (Similarity: {similarity_score:.2f})")
    else:
        context = "No relevant context found."
        print("No strong match found; generating response solely with LLM.")

    prompt = f"Context: {context}\n\nQuestion: {user_question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    outputs = model.generate(**inputs, max_new_tokens=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    match = re.search(r"Answer:\s*(.*?)(?=\s*(?:Question|Context|Answer):|\Z)", generated_answer, re.DOTALL)

    if match:
        generated_answer = match.group(1).strip()
    return generated_answer

def start_gui():
    def handle_user_input(event=None):
        user_question = user_input.get()
        if user_question.lower() == "exit":
            root.quit()
        else:
            add_message(user_question, "user")
            user_input.delete(0, tk.END)

            def process_question():
                bot_response = rag_chatbot(user_question)
                add_message(bot_response, "bot")

            threading.Thread(target=process_question).start()

    def add_message(message, sender):
        frame = tk.Frame(messages_frame, bg="white", pady=5)
        if sender == "user":
            bubble = tk.Label(frame, text=message, bg="#03fc8c", fg="black", wraplength=300, justify="left",
                              anchor="e", padx=10, pady=5)
            bubble.pack(side="left", padx=10)
        else:
            bubble = tk.Label(frame, text=message, bg="#03d3fc", fg="black", wraplength=300, justify="left",
                              anchor="w", padx=10, pady=5)
            bubble.pack(side="left", padx=10)

        frame.pack(anchor="e" if sender == "user" else "w", fill="x", padx=10, pady=2)
        messages_frame.update_idletasks()
        chat_display.yview_moveto(1)

    root = tk.Tk()
    root.title("RAG Chatbot")
    root.geometry("500x600")
    root.resizable(False, True)

    chat_display = tk.Canvas(root, bg="white")
    chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

    scrollbar = tk.Scrollbar(root, command=chat_display.yview)
    scrollbar.grid(row=0, column=2, sticky="ns")
    chat_display.config(yscrollcommand=scrollbar.set)

    messages_frame = tk.Frame(chat_display, bg="white")
    chat_display.create_window((0, 0), window=messages_frame, anchor="nw")

    def configure_scroll_region(event):
        chat_display.configure(scrollregion=chat_display.bbox("all"))

    messages_frame.bind("<Configure>", configure_scroll_region)

    user_input = tk.Entry(root, width=40)
    user_input.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
    user_input.bind("<Return>", handle_user_input)

    send_button = tk.Button(root, text="Send", command=handle_user_input)
    send_button.grid(row=1, column=1, padx=10, pady=10, sticky="ew")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    root.mainloop()

if __name__ == "__main__":
    start_gui()
