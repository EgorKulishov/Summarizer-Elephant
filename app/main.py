# !pip install streamlit transformers torch

import streamlit as st
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Инициализация токенайзера и модели mBART
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-one-mmt")

# Устанавливаем язык, на который настроен mBART
tokenizer.src_lang = "ru_RU"

# Функция сжатия текста
def compress_text(text: str, level: str) -> str:
    """
    Сжимает текст до заданного уровня сжатия.

    Параметры:
    - text (str): Исходный текст для сжатия
    - level (str): Уровень сжатия ("strong", "moderate", "weak")

    Возвращает:
    - str: Сжатый текст
    """
    inputs = tokenizer(text, return_tensors='pt')
    max_length = 128  # Базовая длина

    # Настройка длины текста в зависимости от уровня сжатия
    if level == "strong":
        max_length = 30  # Сильное сжатие: 1-2 предложения
    elif level == "moderate":
        max_length = 60  # Умеренное сжатие: короткий абзац
    elif level == "weak":
        max_length = 90  # Слабое сжатие: чуть больше информации

    # Генерация суммаризации
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,        # Использование beam search для разнообразия
            length_penalty=2.0, # Для более коротких результатов
            no_repeat_ngram_size=3,  # Предотвращает повтор фраз
            early_stopping=True
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Интерфейс приложения Streamlit
def main():
    st.title("Система для сжатия текста на трех уровнях")
    st.write("Выберите уровень сжатия и введите текст для обработки.")

    # Ввод текста
    input_text = st.text_area("Введите текст для сжатия:", height=200)

    # Выбор уровня сжатия
    level = st.selectbox("Выберите уровень сжатия:", ["strong", "moderate", "weak"], index=1)
    level_mapping = {
        "strong": "Сильное сжатие (1-2 предложения)",
        "moderate": "Умеренное сжатие (краткий пересказ)",
        "weak": "Слабое сжатие (абзац)"
    }
    st.write(f"Выбранный уровень сжатия: {level_mapping[level]}")

    # Кнопка для запуска обработки текста
    if st.button("Сжать текст"):
        if input_text.strip():
            # Сжатие текста
            summary = compress_text(input_text, level)
            st.subheader("Результат сжатия:")
            st.write(summary)
        else:
            st.warning("Пожалуйста, введите текст для сжатия.")

if __name__ == "__main__":
    main()

