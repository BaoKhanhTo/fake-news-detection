import re

try:
    from underthesea import word_tokenize
except ImportError:
    # Ham thay the an toan neu thieu thu vien
    def word_tokenize(text, format=None):
        return text.split()

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove special characters
    text = re.sub(r'[^\w\s!?]', ' ', text)
    text = re.sub(r'([!?])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 3. Tokenize (An toan voi ca 2 truong hop)
    try:
        tokens = word_tokenize(text, format="text")
    except:
        tokens = " ".join(text.split())
    
    return tokens

if __name__ == "__main__":
    sample = "Tin nóng: Kiểm tra hệ thống!"
    print(f"Cleaned: {clean_text(sample)}")
