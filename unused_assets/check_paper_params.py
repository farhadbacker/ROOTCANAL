
with open('paper_content.txt', 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

sentences = text.split('.')
keywords = ["distillation", "temperature", "alpha", "teacher", "student", "loss"]

print("-" * 50)
for s in sentences:
    lower_s = s.lower()
    if any(k in lower_s for k in keywords):
        clean_s = s.replace('\n', ' ').strip()
        if len(clean_s) > 10:
            print(f"> {clean_s}\n")
print("-" * 50)
