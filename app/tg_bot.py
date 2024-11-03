import telebot
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Инициализация модели и токенайзера
MODEL_NAME = 'cointegrated/rut5-base-absum'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)  # Установите legacy=False

# Выбор устройства: GPU, если доступен, иначе CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Функция суммаризации с фиксированными значениями длины для каждого уровня
def summarize_text(text: str, max_length: int, min_length: int) -> str:
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

# Создание экземпляра бота
API_TOKEN = 'YOUR_TOKEN'  # Замените на ваш токен
bot = telebot.TeleBot(API_TOKEN)

# Обработка команды /start
@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Добро пожаловать в бот для суммаризации текста! Выберите уровень сжатия.", reply_markup=generate_markup())

# Генерация клавиатуры
def generate_markup():
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True)
    markup.add("Сильное", "Умеренное", "Слабое")
    return markup

# Обработка текстовых сообщений
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    if message.text in ["Сильное", "Умеренное", "Слабое"]:
        bot.send_message(message.chat.id, "Пожалуйста, отправьте текст для суммаризации.")
        bot.register_next_step_handler(message, lambda m: summarize(m, message.text))
    else:
        bot.send_message(message.chat.id, "Пожалуйста, выберите уровень сжатия.")

# Функция для суммаризации
def summarize(message, level):
    text = message.text
    if text.strip():
        if level == "Сильное":
            max_length = 50
            min_length = 25
        elif level == "Умеренное":
            max_length = 100
            min_length = 50
        elif level == "Слабое":
            max_length = 200
            min_length = 100
        
        summary = summarize_text(text, max_length, min_length)
        bot.send_message(message.chat.id, f"Результат суммаризации:\n{summary}")
    else:
        bot.send_message(message.chat.id, "Пожалуйста, введите текст для суммаризации.")

if __name__ == "main":
    bot.polling(none_stop=True)
