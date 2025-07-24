import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import torch

# --- КОНФИГУРАЦИЯ ---
# Укажите путь к папке, где лежит ваш лучший model.safetensors
# Я предполагаю, что вы сохранили лучшую модель в 'piiranha-finetuned-logs/best_model'
MODEL_PATH = "best_model"
DEVICE = 0 if torch.cuda.is_available() else -1 # 0 для GPU, -1 для CPU

# --- CSS Стили для эстетики ---
# Цвета:
# Фон: #2c2f38
# Основной текст: #e0e0e0
# Акцент/заголовки: #6c73ff
# Фон инпутов: #34374f
# Маскированный текст: #ff6565 (красноватый для акцента)

page_style = """
<style>
/* Основной фон и шрифт */
body, .main {
    background-color: #2c2f38;
    color: #e0e0e0;
}
h1, h2, h3 {
    color: #6c73ff;
    font-family: 'Segoe UI', sans-serif;
}

/* Стилизация текстового поля для ввода */
.stTextArea textarea {
    background-color: #34374f;
    color: #e0e0e0;
    border: 1px solid #4a4e69;
    border-radius: 10px;
    font-family: 'Consolas', 'Menlo', 'monospace';
    font-size: 16px;
    padding: 15px;
}

/* Стилизация кнопок */
.stButton>button {
    background-color: #6c73ff;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #5a61e0;
    color: white;
}

/* Стилизация вывода */
.result-box {
    background-color: #34374f;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #6c73ff;
    margin-top: 20px;
    font-family: 'Consolas', 'Menlo', 'monospace';
    font-size: 16px;
    line-height: 1.6;
    white-space: pre-wrap; /* Сохраняет переносы строк */
    word-wrap: break-word; /* Переносит длинные слова */
}

/* Стилизация найденных сущностей */
.entity-table {
    margin-top: 25px;
}
.entity-table th {
    background-color: #4a4e69;
    color: #e0e0e0;
    text-align: left;
}
.entity-table td {
    border-bottom: 1px solid #4a4e69;
}

</style>
"""

# --- Кэширование модели для быстрой загрузки ---
@st.cache_resource
def load_model(path):
    """Загружает модель и токенизатор один раз и кэширует их."""
    try:
        model = AutoModelForTokenClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        # aggregation_strategy="simple" объединяет B- и I- токены в одну сущность (например, B-PASSWORD + I-PASSWORD -> PASSWORD)
        pii_pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
            device=DEVICE,
        )
        return pii_pipeline
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        st.info("Убедитесь, что путь к модели верный и в папке есть все необходимые файлы (config.json, model.safetensors, tokenizer.json и т.д.)")
        return None

# --- Основная функция для маскирования ---
def mask_log(pii_pipeline, text):
    """Принимает текст, находит PII и возвращает замаскированный текст и список найденных сущностей."""
    if not text:
        return "", []

    entities = pii_pipeline(text)
    
    # Сортируем сущности по позиции в тексте в обратном порядке
    # чтобы замена не сбивала индексы последующих сущностей
    sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
    
    masked_text = text
    found_pii = []
    
    for entity in sorted_entities:
        label = entity["entity_group"]
        value = entity["word"]
        start = entity["start"]
        end = entity["end"]
        
        # Создаем маску, например [PASSWORD]
        mask = f"[{label}]"
        
        # Заменяем PII в тексте на маску
        masked_text = masked_text[:start] + mask + masked_text[end:]
        
        # Сохраняем информацию о найденной сущности
        found_pii.append({"Тип данных": label, "Найденное значение": value})
        
    # Возвращаем в правильном порядке
    return masked_text, sorted(found_pii, key=lambda x: text.find(x["Найденное значение"]))


# --- ИНТЕРФЕЙС STREAMLIT ---

# Применяем наши стили
st.markdown(page_style, unsafe_allow_html=True)

# Загружаем модель
pii_pipeline = load_model(MODEL_PATH)

# Заголовок
st.title("Детектор персональных данных в логах")
st.markdown("Введите текст лога для анализа и маскировки PII.")

# Поле для ввода текста
log_input = st.text_area(
    "📝 Вставьте ваш лог сюда:",
    height=250,
    placeholder="Например: 2025-05-22 13:42:14 ERROR User request processed, email franklinjames@example.net, card 4851961064628792322..."
)

# Кнопка для запуска анализа
if st.button("🔍 Проанализировать и замаскировать"):
    if pii_pipeline and log_input:
        with st.spinner("Анализирую..."):
            masked_text, found_pii = mask_log(pii_pipeline, log_input)
            
            st.subheader("🔒 Замаскированный лог")
            st.markdown(f'<div class="result-box">{masked_text}</div>', unsafe_allow_html=True)
            
            if found_pii:
                st.subheader("🔎 Найденные и скрытые данные")
                # Выводим таблицу с найденными данными
                st.dataframe(found_pii, use_container_width=True, hide_index=True)
            else:
                st.success("Персональные данные в этом логе не найдены.")
    elif not log_input:
        st.warning("Пожалуйста, введите текст для анализа.")

st.markdown("---")
st.info("Это приложение использует модель, дообученную для распознавания именованных сущностей (PII).")