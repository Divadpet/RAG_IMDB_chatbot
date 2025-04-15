import tkinter as tk
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import numpy as np
import json
import os
import re

path = "Tekarukite/UHK-IMDB-Dataaset2"

dataset = load_dataset(path=path, split="train")

assert "question" in dataset.column_names and "answer" in dataset.column_names, \
    "Dataset musí mít sloupec 'question' a 'answer'."

questions = dataset["question"]
answers = dataset["answer"]

vectorizer = TfidfVectorizer(stop_words='english')
question_vectors = vectorizer.fit_transform(questions)

def split_into_chunks(text):
    chunks = re.split(r'[,.&]| and ', text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def retrieve_answer(user_question, similarity_weight=0.5, word_count_weight=0.5, score_threshold=0.7):
    chunks = split_into_chunks(user_question)

    best_combined_score = 0
    best_answer = None

    for chunk in chunks:
        user_vector = vectorizer.transform([chunk])
        similarities = cosine_similarity(user_vector, question_vectors).flatten()

        user_word_count = len(chunk.split())

        combined_scores = []
        for idx, sim_score in enumerate(similarities):
            match_word_count = len(questions[idx].split())
            word_count_similarity = 1 - abs(user_word_count - match_word_count) / max(user_word_count, match_word_count)
            
            combined_score = (similarity_weight * sim_score) + (word_count_weight * word_count_similarity)
            combined_scores.append(combined_score)

        chunk_best_index = np.argmax(combined_scores)
        chunk_best_score = combined_scores[chunk_best_index]

        if chunk_best_score > best_combined_score:
            best_combined_score = chunk_best_score
            best_answer = answers[chunk_best_index]

    if best_combined_score >= score_threshold:
        return best_answer, best_combined_score
    else:
        return None, best_combined_score


token = ""
    
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token=token)
pipeline = pipeline(task="text-generation", model=model_name, device_map="auto", token=token)
device = model.device

'''
import time
questions_list = [
"Who is the director of The Shawshank Redemption?",  # Otázka z datasetu
"What is the plot of Inception?",  # Otázka z datasetu
"List the top cast of The Matrix",  # Otázka z datasetu
"What genre is The Dark Knight?",  # Otázka z datasetu
"When was The Godfather released?",  # Otázka z datasetu
"How much did Jurassic Park cost to make?",  # Otázka z datasetu
"What is the IMDB rating of Pulp Fiction?",  # Otázka z datasetu
"What are the countries of origin for Titanic?",  # Otázka z datasetu
"What is the release date of Avatar?",  # Otázka z datasetu
"Provide the storyline of The Lord of the Rings",  # Otázka z datasetu
"What is the storyline of The Matrix?",  # Jinak formulovaná otázka
"Can you list the main actors in The Shawshank Redemption?",  # Jinak formulovaná otázka
"Who acted in The Godfather?",  # Jinak formulovaná otázka
"Tell me the plot of Jurassic Park",  # Jinak formulovaná otázka
"How many reviews does Pulp Fiction have on IMDB?",  # Jinak formulovaná otázka
"When was Fight Club released?",  # Jinak formulovaná otázka
"What language is spoken in The Lion King?",  # Jinak formulovaná otázka
"Who directed Star Wars: A New Hope?",  # Jinak formulovaná otázka
"What is the genre of Gladiator?",  # Jinak formulovaná otázka
"How popular is Avatar on IMDB?"  # Jinak formulovaná otázka
]

# Funkce pro zodpovězení všech otázek
def answer_all_questions():
    for question in questions_list:
        start_time = time.time()  # Start časového měření
        
        print(f"Question: {question}")
        answer = rag_chatbot(question)
        
        end_time = time.time()  # Konec měření
        elapsed_time = end_time - start_time  # Výpočet uplynulého času
        
        print(f"Answer: {answer}")
        print(f"Time taken: {elapsed_time:.2f} seconds")  # Vytiskne čas v sekundách
        print("-" * 50)
'''
current_chat_id = None
chat = [{"role": "system", "content": "You are a chatbot that answers movie questions using provided context."}]
CHAT_HISTORY_FILE = "saved_chats.json"

def generate_chat_id():
    all_chats = load_chats()
    if not all_chats:
        return "1"
    else:
        existing_ids = [int(cid) for cid in all_chats.keys() if cid.isdigit()]
        if not existing_ids:
            return "1"
        return str(max(existing_ids) + 1)

def save_chat():
    global current_chat_id

    if not os.path.exists(CHAT_HISTORY_FILE):
        all_chats = {}
    else:
        with open(CHAT_HISTORY_FILE, "r") as f:
            all_chats = json.load(f)

    if current_chat_id:
        if current_chat_id in all_chats:
            all_chats[current_chat_id] = chat
        else:
            print(f"Chat with ID {current_chat_id} not found. Creating a new chat.")
            all_chats[current_chat_id] = chat
    else:
        current_chat_id = generate_chat_id()
        all_chats[current_chat_id] = chat

    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(all_chats, f, indent=2)
        

def load_chats():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}
    with open(CHAT_HISTORY_FILE, "r") as f:
        return json.load(f)

def rag_chatbot(user_question):
    global chat
    
    retrieved_answer, similarity_score = retrieve_answer(user_question)

    if retrieved_answer:
        context = retrieved_answer
        print(f"Retrieved answer used (Similarity: {similarity_score:.2f})")
    else:
        context = "No relevant context found."
        print("No strong match found; generating response solely with LLM.")

    chat.append({"role": "user", "content": user_question})
    chat.append({"role": "system", "content": context})

    generated_answer = pipeline(chat, max_new_tokens=512)
    generated_answer = generated_answer[0]["generated_text"][-1]["content"]

    chat.append({"role": "assistant", "content": generated_answer})

    return generated_answer

def set_current_chat_id(cid):
    global current_chat_id
    current_chat_id = cid

def start_gui():
    def handle_user_input(event=None):
        user_question = user_input.get()
        if not user_question.strip():
            return
        add_message(user_question, "user")
        user_input.delete(0, tk.END)

        def process_question():
            bot_response = rag_chatbot(user_question)
            add_message(bot_response, "bot")
        threading.Thread(target=process_question).start()

    def add_message(message, sender):
        frame = tk.Frame(messages_frame, bg="white", pady=5)

        bubble_color = "#DCF8C6" if sender == "user" else "#E6E6E6"
        bubble = tk.Label(
            frame,
            text=message,
            bg=bubble_color,
            fg="black",
            font=("Helvetica", 12),
            wraplength=300,
            justify="left",
            anchor="w",
            padx=12,
            pady=8,
            bd=0,
            relief="solid",
        )

        bubble.pack(anchor="w", padx=10, pady=5)
        frame.pack(anchor="w", fill="x")
        refresh_chat_buttons()

    def load_chat(chat_id):
        global chat
        all_chats = load_chats()

        if chat_id in all_chats:
            for widget in messages_frame.winfo_children():
                widget.destroy()

            chat = all_chats[chat_id]

            for message in chat:
                if message["role"] in ["user", "assistant"]:
                    sender = "user" if message["role"] == "user" else "bot"
                    add_message(message["content"], sender)

            chat_display.yview_moveto(0)

    root = tk.Tk()
    root.title("Movie Chatbot (Built with Llama)")
    root.geometry("700x650")
    root.configure(bg="white")

    sidebar = tk.Frame(root, width=200, bg="#eeeeee")
    sidebar.grid(row=0, column=0, rowspan=2, sticky="ns")

    chats_list_frame = tk.Frame(sidebar, bg="#eeeeee")
    chats_list_frame.pack(fill="both", expand=True)

    def refresh_chat_buttons():
        for widget in chats_list_frame.winfo_children():
            widget.destroy()

        all_chats = load_chats()

        for chat_id in all_chats:
            chat_row = tk.Frame(chats_list_frame, bg="#eeeeee")
            chat_row.pack(fill="x", pady=2)

            chat_button = tk.Button(
                chat_row,
                text=f"Chat {chat_id}",
                command=lambda cid=chat_id: (set_current_chat_id(cid), load_chat(cid)),
                font=("Helvetica", 10),
                bg="#f0f0f0",
                relief="ridge"
            )
            chat_button.pack(side="left", fill="x", expand=True)

            delete_button = tk.Button(
                chat_row,
                text="❌",
                command=lambda cid=chat_id: delete_chat(cid),
                font=("Helvetica", 10),
                bg="#ff4d4d",
                fg="white",
                relief="ridge"
            )
            delete_button.pack(side="right", padx=5)

    def delete_chat(chat_id):
        if not os.path.exists(CHAT_HISTORY_FILE):
            return

        with open(CHAT_HISTORY_FILE, "r") as f:
            all_chats = json.load(f)

        if chat_id in all_chats:
            del all_chats[chat_id]
            if chat_id == current_chat_id:
                start_new_chat()

            with open(CHAT_HISTORY_FILE, "w") as f:
                json.dump(all_chats, f, indent=2)

        refresh_chat_buttons()

    def start_new_chat():
        global chat
        global current_chat_id
        chat = [{"role": "system", "content": "You are a chatbot that answers movie questions."}]
        current_chat_id=None
        for widget in messages_frame.winfo_children():
            widget.destroy()

        save_chat()
        refresh_chat_buttons()
    
    new_chat_button = tk.Button(
        sidebar,
        text="New Chat",
        command=start_new_chat,
        font=("Helvetica", 12),
        bg="#4CAF50",
        fg="white",
        padx=10,
        pady=10
    )
    new_chat_button.pack(fill="x", pady=10)

    refresh_chat_buttons()

    chat_display = tk.Canvas(root, bg="white", highlightthickness=0)
    chat_display.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

    scrollbar = tk.Scrollbar(root, command=chat_display.yview)
    scrollbar.grid(row=0, column=3, sticky="ns")
    chat_display.config(yscrollcommand=scrollbar.set)

    messages_frame = tk.Frame(chat_display, bg="white")
    chat_display.create_window((0, 0), window=messages_frame, anchor="nw")

    def configure_scroll_region(event):
        chat_display.configure(scrollregion=chat_display.bbox("all"))

    messages_frame.bind("<Configure>", configure_scroll_region)

    user_input = tk.Entry(root, font=("Helvetica", 14))
    user_input.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
    user_input.bind("<Return>", handle_user_input)

    send_button = tk.Button(
        root,
        text="Send",
        command=handle_user_input,
        font=("Helvetica", 14),
        bg="#4CAF50",
        fg="white",
        activebackground="#45A049",
        bd=0,
        padx=20,
        pady=10,
        relief="ridge",
        cursor="hand2"
    )
    send_button.grid(row=1, column=2, padx=(0,10), pady=10, sticky="ew")

    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    root.mainloop()

if __name__ == "__main__":
    #answer_all_questions()
    start_gui()