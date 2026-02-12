# TLS_client.py
import socket
import json
import numpy as np
import base64
import struct
import time
from typing import Tuple, Optional, Dict, Any, List
from datetime import datetime

class PerlinMeshClient:
    """Клиент для генерации перлин-меша и булевых операций через C++ сервер"""

    def __init__(self, host='192.168.123.53', port=9999, timeout=180.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.metrics = {}

    # -----------------------------------------------------------------
    #  Внутренние вспомогательные методы
    # -----------------------------------------------------------------
    def _recv_exact(self, sock, num_bytes):
        """Получает точное количество байт"""
        data = b''
        start_time = time.time()
        while len(data) < num_bytes:
            try:
                remaining = num_bytes - len(data)
                chunk_size = min(65536, remaining)
                chunk = sock.recv(chunk_size)
                if not chunk:
                    return None
                data += chunk
            except socket.timeout:
                elapsed = time.time() - start_time
                print(f"Таймаут при получении данных: получено {len(data)}/{num_bytes} байт за {elapsed:.1f} сек")
                return None
        return data

    def _encode_mesh(self, vertices: np.ndarray, indices: np.ndarray) -> Dict[str, str]:
        """Кодирует вершины и индексы в Base64 для передачи"""
        v_flat = vertices.astype(np.float32).flatten()
        i_flat = indices.astype(np.uint32).flatten()
        v_bytes = v_flat.tobytes()
        i_bytes = i_flat.tobytes()
        return {
            "vertices": base64.b64encode(v_bytes).decode('ascii'),
            "indices": base64.b64encode(i_bytes).decode('ascii')
        }

    def _decode_mesh(self, vertices_b64: str, indices_b64: str) -> Tuple[np.ndarray, np.ndarray]:
        """Декодирует Base64 обратно в numpy массивы"""
        v_bytes = base64.b64decode(vertices_b64)
        i_bytes = base64.b64decode(indices_b64)
        vertices = np.frombuffer(v_bytes, dtype=np.float32).reshape(-1, 3)
        indices = np.frombuffer(i_bytes, dtype=np.uint32).reshape(-1, 3)
        return vertices, indices

    def _send_request(self, request_dict: Dict) -> Optional[Dict]:
        """Базовый метод отправки запроса и получения ответа"""
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            request_json = json.dumps(request_dict, separators=(',', ':'), ensure_ascii=False)
            request_bytes = request_json.encode('utf-8')

            # Отправляем размер (4 байта, big-endian)
            size_bytes = struct.pack('>I', len(request_bytes))
            sock.sendall(size_bytes)
            sock.sendall(request_bytes)

            # Получаем размер ответа
            size_resp = self._recv_exact(sock, 4)
            if not size_resp:
                return None
            response_size = struct.unpack('>I', size_resp)[0]

            # Получаем данные
            response_bytes = self._recv_exact(sock, response_size)
            if not response_bytes:
                return None

            response_str = response_bytes.decode('utf-8', errors='ignore')
            return json.loads(response_str)

        except Exception as e:
            print(f"Ошибка отправки запроса: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            if sock:
                sock.close()

    # -----------------------------------------------------------------
    #  Генерация перлин-меша (восстановленный метод)
    # -----------------------------------------------------------------
    def send_perlin_request(
        self,
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
        vertices_before: List[Tuple[float, float, float]],
        texcoords_before: List[Tuple[float, float]]
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Генерирует перлин-меш на удалённом сервере.

        Args:
            grid_size: размер сетки (grid_size x grid_size)
            size_x, size_y, size_z: габариты меша
            base_z: базовая Z-координата
            noise_scale: масштаб шума
            octaves: количество октав
            persistence: персистенция
            lacunarity: лакунарность
            seed: зерно ГПСЧ
            texture_repeatX, texture_repeatY: повтор текстурных координат
            strength: сила смещения по карте высот
            height_array: 2D массив карты высот (float32)
            vertices_before: список кортежей (x, y, z) исходных вершин до смещения
            texcoords_before: список кортежей (u, v) исходных UV

        Returns:
            Кортеж (vertices, normals, texcoords) – numpy массивы float32,
            или None при ошибке.
        """
        # Подготовка больших данных
        # height_array: (H, W) -> байты float32
        height_bytes = height_array.astype(np.float32).tobytes()
        height_b64 = base64.b64encode(height_bytes).decode('ascii')

        # vertices_before -> массив (N,3) float32
        v_before_arr = np.array(vertices_before, dtype=np.float32)
        v_before_bytes = v_before_arr.tobytes()
        v_before_b64 = base64.b64encode(v_before_bytes).decode('ascii')

        # texcoords_before -> массив (N,2) float32
        t_before_arr = np.array(texcoords_before, dtype=np.float32)
        t_before_bytes = t_before_arr.tobytes()
        t_before_b64 = base64.b64encode(t_before_bytes).decode('ascii')

        # Формируем запрос
        request = {
            "type": "perlin",  # ожидаемый сервером тип операции
            "grid_size": grid_size,
            "size_x": size_x,
            "size_y": size_y,
            "size_z": size_z,
            "base_z": base_z,
            "noise_scale": noise_scale,
            "octaves": octaves,
            "persistence": persistence,
            "lacunarity": lacunarity,
            "seed": seed,
            "texture_repeatX": texture_repeatX,
            "texture_repeatY": texture_repeatY,
            "strength": strength,
            "height_array": height_b64,
            "vertices_before": v_before_b64,
            "texcoords_before": t_before_b64
        }

        start_time = time.time()
        response = self._send_request(request)
        if not response:
            print("Нет ответа от сервера при генерации перлин-меша")
            return None

        if response.get("status") != "success":
            error_msg = response.get("error", "Unknown error")
            print(f"Ошибка генерации перлин-меша: {error_msg}")
            return None

        # Декодируем результат
        try:
            result_data = response.get("result", {})
            vertices_b64 = result_data.get("vertices")
            normals_b64 = result_data.get("normals")
            texcoords_b64 = result_data.get("texcoords")

            if not vertices_b64 or not normals_b64 or not texcoords_b64:
                print("Неполные данные в ответе сервера")
                return None

            # Декодируем base64 -> numpy
            vertices_bytes = base64.b64decode(vertices_b64)
            normals_bytes = base64.b64decode(normals_b64)
            texcoords_bytes = base64.b64decode(texcoords_b64)

            vertices = np.frombuffer(vertices_bytes, dtype=np.float32).reshape(-1, 3)
            normals = np.frombuffer(normals_bytes, dtype=np.float32).reshape(-1, 3)
            texcoords = np.frombuffer(texcoords_bytes, dtype=np.float32).reshape(-1, 2)

            # Сохраняем метрики
            metrics = response.get("metrics", {})
            total_time = time.time() - start_time
            self.metrics = {
                'timestamp': datetime.now().isoformat(),
                'operation': 'perlin',
                'grid_size': grid_size,
                'server_time_ms': metrics.get('total_time_ms', 0),
                'client_time_sec': total_time,
                'vertex_count': vertices.shape[0]
            }

            return vertices, normals, texcoords

        except Exception as e:
            print(f"Ошибка декодирования результата перлин-меша: {e}")
            import traceback
            traceback.print_exc()
            return None

    # -----------------------------------------------------------------
    #  Методы для булевых операций (без изменений)
    # -----------------------------------------------------------------
    def upload_target_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> Optional[str]:
        """Загружает целевую сетку на сервер и получает session_id."""
        mesh_encoded = self._encode_mesh(vertices, faces)
        request = {
            "type": "upload_target_mesh",
            "mesh": mesh_encoded
        }

        start_time = time.time()
        response = self._send_request(request)
        if not response:
            return None

        if response.get("status") != "success":
            print(f"Ошибка загрузки целевой сетки: {response.get('error', 'Unknown error')}")
            return None

        session_id = response.get("session_id")
        metrics = response.get("metrics", {})

        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'upload_target_mesh',
            'total_time': metrics.get('total_time_ms', 0) / 1000.0,
            'vertex_count': metrics.get('vertex_count', 0),
            'triangle_count': metrics.get('triangle_count', 0),
            'session_id': session_id
        }

        print(f"Целевая сетка загружена. Сессия: {session_id}")
        return session_id

    def boolean_difference_with_session(self, session_id: str,
                                        perlin_vertices: np.ndarray,
                                        perlin_faces: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Выполняет булеву разность target - perlin, используя сохранённую на сервере целевую сетку."""
        mesh2_encoded = self._encode_mesh(perlin_vertices, perlin_faces)
        request = {
            "type": "boolean_difference_with_session",
            "session_id": session_id,
            "mesh2": mesh2_encoded
        }

        start_time = time.time()
        response = self._send_request(request)
        if not response:
            return None

        if response.get("status") != "success":
            print(f"Ошибка булевой разности: {response.get('error', 'Unknown error')}")
            return None

        result_data = response.get("result", {})
        metrics = response.get("metrics", {})

        if not result_data.get("vertices") or not result_data.get("indices"):
            return None, None, metrics

        vertices, faces = self._decode_mesh(result_data["vertices"], result_data["indices"])
        total_time = time.time() - start_time
        metrics['client_total_time_sec'] = total_time

        self.metrics = {
            'timestamp': datetime.now().isoformat(),
            'operation': 'boolean_difference_with_session',
            'session_id': session_id,
            'server_metrics': metrics,
            'client_time_sec': total_time
        }

        return vertices, faces, metrics

    def boolean_difference(self,
                           mesh1_vertices: np.ndarray, mesh1_faces: np.ndarray,
                           mesh2_vertices: np.ndarray, mesh2_faces: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Выполняет булеву разность mesh1 - mesh2 (однократная операция, без сессии)."""
        mesh1_encoded = self._encode_mesh(mesh1_vertices, mesh1_faces)
        mesh2_encoded = self._encode_mesh(mesh2_vertices, mesh2_faces)
        request = {
            "type": "boolean_difference",
            "mesh1": mesh1_encoded,
            "mesh2": mesh2_encoded
        }

        start_time = time.time()
        response = self._send_request(request)
        if not response:
            return None

        if response.get("status") != "success":
            print(f"Ошибка булевой разности: {response.get('error', 'Unknown error')}")
            return None

        result_data = response.get("result", {})
        metrics = response.get("metrics", {})

        if not result_data.get("vertices") or not result_data.get("indices"):
            return None, None, metrics

        vertices, faces = self._decode_mesh(result_data["vertices"], result_data["indices"])
        total_time = time.time() - start_time
        metrics['client_total_time_sec'] = total_time
        return vertices, faces, metrics

    def get_metrics(self):
        """Возвращает последние собранные метрики"""
        return self.metrics