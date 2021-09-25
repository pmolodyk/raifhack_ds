# Описание
* train.py - скрипт, который обучает модель и сохраняет ее
* predict.py - скрипт, который делает предсказание на отложенной тестовой выборке

# Запуск
## Вариант с requirements.txt
<ol>
    <li> убедиться, что у вас стоит python3.6 или выше </li>
    <li> установить зависимости:
    
    pip install -r requirements.txt 
</li>
    <li> запустить обучение

    python3 train.py --train_data df_train.csv.zip --val_data df_val.csv.zip
</li>
    <li> запустить предикт
    
    python predict.py --test_data test.csv --model_path model --output ans.csv
</li>
</ol>

        `
