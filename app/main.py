import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Инициализация модели и токенайзера
MODEL_NAME = 'cointegrated/rut5-base-absum'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

# Выбор устройства: GPU, если доступен, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Функция суммаризации с фиксированными значениями длины для каждого уровня
def summarize_text(text: str, max_length: int, min_length: int) -> str:
    """
    Сжимает текст на основе фиксированной максимальной и минимальной длины.

    Параметры:
    - text (str): Исходный текст для суммаризации
    - max_length (int): Максимальная длина суммаризации в токенах
    - min_length (int): Минимальная длина суммаризации в токенах

    Возвращает:
    - str: Сжатый текст
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Интерфейс Streamlit
def main():
    st.title("Суммаризация текста с фиксированными уровнями сжатия")
    st.write("Введите текст и выберите желаемый уровень сжатия.")

    # Ввод текста
    input_text = st.text_area("Введите текст для суммаризации:", height=200)

    # Выбор уровня сжатия
    level = st.selectbox("Выберите уровень сжатия:", ["Сильное", "Умеренное", "Слабое"], index=1)
    
    # Установка параметров для каждого уровня
    if level == "Сильное":
        max_length = 50    # Сильное сжатие (примерно 1-2 предложения)
        min_length = 25
    elif level == "Умеренное":
        max_length = 100   # Умеренное сжатие (краткий пересказ)
        min_length = 50
    elif level == "Слабое":
        max_length = 200   # Слабое сжатие (краткий абзац)
        min_length = 100

    # Кнопка для выполнения суммаризации
    if st.button("Сжать текст"):
        if input_text.strip():
            summary = summarize_text(input_text, max_length, min_length)
            st.subheader("Результат суммаризации:")
            st.write(summary)
        else:
            st.warning("Пожалуйста, введите текст для суммаризации.")

if __name__ == "__main__":
    main()
