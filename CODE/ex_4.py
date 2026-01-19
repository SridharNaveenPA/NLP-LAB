from transformers import MarianMTModel, MarianTokenizer

source_language = "en"
target_language = "fr"
model_name = "Helsinki-NLP/opus-mt-en-fr"

tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

text = input("Enter text to translate: ")

tokens = tokenizer(text, return_tensors="pt", padding=True)

translated_tokens = model.generate(**tokens)

translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("\nTranslated Text:")
print(translated_text)
