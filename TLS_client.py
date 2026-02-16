import requests
import numpy as np
import base64
import json
from typing import Tuple, Optional, Union

class TLS_client:
    """
    Клиент для взаимодействия с сервером перлин-генерации и булевых операций через REST API.
    """

    def __init__(self, host='192.168.123.53', port=9999, timeout=300.0):
        """
        :param host: IP-адрес сервера
        :param port: порт сервера
        :param timeout: таймаут на весь запрос (в секундах)
        """
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> dict:
        """
        Внутренний метод для отправки POST-запроса.
        Возвращает распарсенный JSON-ответ или выбрасывает исключение.
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()  # выбросит исключение при HTTP-ошибке
            return resp.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Таймаут при обращении к серверу ({self.timeout} сек)")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ошибка подключения к серверу {url}: {e}")
        except requests.exceptions.HTTPError as e:
            # Попытаемся извлечь JSON с описанием ошибки от сервера
            try:
                error_data = e.response.json()
                msg = error_data.get("error", str(e))
            except:
                msg = str(e)
            raise RuntimeError(f"Сервер вернул ошибку: {msg}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при запросе: {e}")

    def send_perlin_request(self,
                            grid_size: int,
                            size_x: float,
                            size_y: float,
                            size_z: float,
                            base_z: float,
                            noise_scale: float,
                            octaves: int,
                            persistence: float,
                            lacunarity: float,
                            seed: int,
                            texture_repeatX: float,
                            texture_repeatY: float,
                            strength: float,
                            height_array: np.ndarray,
                            vertices_before: list,
                            texcoords_before: list) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Запрос на генерацию перлин-меша.
        Возвращает кортеж (vertices, normals, texcoords) как numpy массивы.
        """
        # Преобразуем height_array в base64
        tex_height, tex_width = height_array.shape
        height_normalized = (height_array * 255).astype(np.uint8)
        height_map_b64 = base64.b64encode(height_normalized.tobytes()).decode('ascii')

        # Формируем payload
        payload = {
            "type": "generate_perlin_mesh",  # сервер ожидает это поле (можно не отправлять, если endpoint отдельный, но оставим для совместимости)
            "grid_size": grid_size,
            "size_x": float(size_x),
            "size_y": float(size_y),
            "size_z": float(size_z),
            "base_z": float(base_z),
            "noise_scale": float(noise_scale),
            "octaves": octaves,
            "persistence": float(persistence),
            "lacunarity": float(lacunarity),
            "seed": seed,
            "texture_repeatX": float(texture_repeatX),
            "texture_repeatY": float(texture_repeatY),
            "strength": float(strength),
            "tex_width": tex_width,
            "tex_height": tex_height,
            "height_map": height_map_b64,
            # vertices_before и texcoords_before игнорируются сервером, но оставляем для совместимости API
        }

        response = self._post("generate_perlin_mesh", payload)

        if response.get("status") != "success":
            raise RuntimeError(f"Ошибка сервера: {response.get('error', 'Неизвестная ошибка')}")

        result = response.get("result", {})
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                raise RuntimeError("Ошибка парсинга result")

        try:
            vertices = np.array(result["vertices"], dtype=np.float32).reshape(-1, 3)
            normals = np.array(result["normals"], dtype=np.float32).reshape(-1, 3)
            texcoords = np.array(result["texcoords"], dtype=np.float32).reshape(-1, 2)
        except (KeyError, ValueError) as e:
            raise RuntimeError(f"Ошибка преобразования данных: {e}")

        return vertices, normals, texcoords

    def send_boolean_request(self,
                             mesh1_vertices: np.ndarray,
                             mesh1_triangles: np.ndarray,
                             mesh2_vertices: np.ndarray,
                             mesh2_triangles: np.ndarray,
                             return_volume_only: bool = False) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        """
        Запрос на булеву разность двух сеток.
        Если return_volume_only=True, возвращает объём (float).
        Иначе возвращает кортеж (vertices, triangles) как numpy массивы.
        """
        # Приведение типов и кодирование в base64
        v1 = np.asarray(mesh1_vertices, dtype=np.float32)
        t1 = np.asarray(mesh1_triangles, dtype=np.uint32)
        v2 = np.asarray(mesh2_vertices, dtype=np.float32)
        t2 = np.asarray(mesh2_triangles, dtype=np.uint32)

        payload = {
            "type": "boolean_difference",  # опционально
            "mesh1_vertices": base64.b64encode(v1.tobytes()).decode('ascii'),
            "mesh1_triangles": base64.b64encode(t1.tobytes()).decode('ascii'),
            "mesh2_vertices": base64.b64encode(v2.tobytes()).decode('ascii'),
            "mesh2_triangles": base64.b64encode(t2.tobytes()).decode('ascii'),
            "return_volume_only": "true" if return_volume_only else "false"
        }

        response = self._post("boolean_difference", payload)

        if response.get("status") != "success":
            raise RuntimeError(f"Ошибка сервера: {response.get('error', 'Неизвестная ошибка')}")

        if return_volume_only:
            try:
                return float(response["volume"])
            except (KeyError, ValueError):
                raise RuntimeError("Сервер не вернул объём")
        else:
            try:
                verts_b64 = response["vertices"]
                tris_b64 = response["triangles"]
                verts = np.frombuffer(base64.b64decode(verts_b64), dtype=np.float32).reshape(-1, 3)
                tris = np.frombuffer(base64.b64decode(tris_b64), dtype=np.uint32).reshape(-1, 3)
                return verts, tris
            except (KeyError, ValueError) as e:
                raise RuntimeError(f"Ошибка декодирования результата: {e}")