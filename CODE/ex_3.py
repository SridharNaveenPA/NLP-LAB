import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

# Define intents and responses
responses = {
    "greeting": ["Hello!", "Hi there!", "Hey! How can I help you?"],
    "goodbye": ["Bye!", "See you later!", "Goodbye!"],
    "thanks": ["You're welcome!", "Happy to help!", "No problem!"],
    "name": ["I am a simple chatbot.", "You can call me ChatBot 🤖"],
    "help": ["I can answer simple questions.", "Try asking about me."],
    "default": ["Sorry, I didn't understand that."]
}

# Keywords for intent detection
intents = {
    "greeting": ["hi", "hello", "hey"],
    "goodbye": ["bye", "exit", "quit"],
    "thanks": ["thanks", "thank"],
    "name": ["name", "who"],
    "help": ["help", "support"]
}

def get_intent(user_input):
    tokens = word_tokenize(user_input.lower())
    for intent, keywords in intents.items():
        for word in tokens:
            if word in keywords:
                return intent
    return "default"

def chatbot():
    print("ChatBot 🤖: Hello! Type 'bye' to exit.")
    while True:
        user_input = input("You: ")
        intent = get_intent(user_input)
        response = responses[intent][0]
        print("ChatBot 🤖:", response)
        if intent == "goodbye":
            break

if __name__ == "__main__":
    chatbot()
