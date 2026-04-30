import requests
import os

server_url = "http://192.168.123.53:9999/upload"  # замените IP/порт при необходимости

local_folder = ""

# Имена файлов для загрузки
files_to_upload = ["1f1fd93a-d246-4b86-af9a-0438627827da_20260327102147.json", "1f1fd93a-d246-4b86-af9a-0438627827da_20260327102147_nonlpr.jpg", "1f1fd93a-d246-4b86-af9a-0438627827da_20260327102147.ply"]

for filename in files_to_upload:
    file_path = os.path.join(local_folder, filename)
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден, пропускаем.")
        continue

    with open(file_path, "rb") as f:
        file_data = f.read()

    headers = {"X-Filename": filename}
    response = requests.post(server_url, data=file_data, headers=headers)

    if response.status_code == 200:
        print(f"✅ {filename} успешно загружен.")
    else:
        print(f"❌ Ошибка загрузки {filename}: {response.status_code} - {response.text}")