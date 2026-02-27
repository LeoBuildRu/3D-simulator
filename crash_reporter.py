import sys
import os
import traceback
import requests
from datetime import datetime
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPixmap, QScreen
from PyQt5.QtCore import PYQT_VERSION_STR

# Флаг для предотвращения рекурсивных вызовов
_inside_excepthook = False

class TelegramCrashReporter:
    """Отправляет информацию об ошибке в Telegram."""
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/"

    def send_message(self, text, parse_mode='HTML'):
        """Отправляет текстовое сообщение."""
        url = self.api_url + "sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': text,
            'parse_mode': parse_mode
        }
        try:
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")

    def send_screenshot(self, caption=''):
        """Делает скриншот главного окна и отправляет как фото."""
        try:
            screen = QApplication.primaryScreen()
            if screen:
                pixmap = screen.grabWindow(0)  # весь рабочий стол
                screenshot_path = 'crash_screenshot.png'
                pixmap.save(screenshot_path, 'PNG')

                url = self.api_url + "sendPhoto"
                with open(screenshot_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {'chat_id': self.chat_id, 'caption': caption}
                    requests.post(url, data=data, files=files, timeout=10)
                os.remove(screenshot_path)
        except Exception as e:
            print(f"Failed to send screenshot: {e}")

    def report_exception(self, exc_type, exc_value, exc_traceback):
        """Формирует и отправляет отчёт об исключении."""
        global _inside_excepthook
        if _inside_excepthook:
            # Предотвращаем рекурсию
            return
        _inside_excepthook = True

        try:
            # Собираем информацию об ошибке
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            tb_text = ''.join(tb_lines)

            # Информация о системе
            import platform
            system_info = (
                f"OS: {platform.system()} {platform.release()}\n"
                f"Python: {sys.version.split()[0]}\n"
                f"PyQt: {PYQT_VERSION_STR}\n"
            )

            message = (
                f"<b>🚨 CRASH REPORT</b>\n"
                f"<b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"<b>System:</b>\n{system_info}\n"
                f"<b>Exception:</b>\n<pre>{tb_text[-3500:]}</pre>"
            )

            # Отправляем текст
            self.send_message(message)

            # Опционально: отправляем скриншот
            # self.send_screenshot(caption="Скриншот в момент ошибки")
        finally:
            _inside_excepthook = False