import requests
import urllib.parse
import numpy as np
import base64
import json
from typing import Tuple, Optional, Union, List

class TLS_client:
    """
    Клиент для взаимодействия с сервером перлин-генерации и булевых операций через REST API.
    """

    def __init__(self, host='192.168.123.53', port=9999, timeout=300.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout

    def _post(self, endpoint: str, payload: dict) -> dict:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Таймаут при обращении к серверу ({self.timeout} сек)")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ошибка подключения к серверу {url}: {e}")
        except requests.exceptions.HTTPError as e:
            try:
                error_data = e.response.json()
                msg = error_data.get("error", str(e))
            except:
                msg = str(e)
            raise RuntimeError(f"Сервер вернул ошибку: {msg}")
        except requests.exceptions.JSONDecodeError as e:
            preview = resp.text[:500] + "..." if len(resp.text) > 500 else resp.text
            raise RuntimeError(
                f"Сервер вернул невалидный JSON: {e}\n"
                f"Первые 500 символов ответа:\n{preview}"
            )
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при запросе: {e}")

    def _get(self, endpoint: str, params: dict = None) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Таймаут при обращении к серверу ({self.timeout} сек)")
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(f"Ошибка подключения к серверу {url}: {e}")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Сервер вернул ошибку: {e}")
        except Exception as e:
            raise RuntimeError(f"Неизвестная ошибка при запросе: {e}")

    # ====================== СУЩЕСТВУЮЩИЕ МЕТОДЫ ======================
    def get_verified_models(self) -> list:
        response = self._post("get_verified_models", {})
        if response.get("status") != "success":
            raise RuntimeError(f"Ошибка сервера: {response.get('error', 'Неизвестная ошибка')}")
        return response.get("files", [])

    def download_file(self, filename: str, local_path: str) -> None:
        url = f"{self.base_url}/download?file={urllib.parse.quote(filename)}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(resp.content)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"Таймаут при скачивании файла {filename}")
        except Exception as e:
            raise RuntimeError(f"Ошибка скачивания файла {filename}: {e}")

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
        tex_height, tex_width = height_array.shape
        height_normalized = (height_array * 255).astype(np.uint8)
        height_map_b64 = base64.b64encode(height_normalized.tobytes()).decode('ascii')
        payload = {
            "type": "generate_perlin_mesh",
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
    
    def generate_landscape(self,
                        # Геометрия
                        size: float = 10.0,
                        subdivisions: int = 64,
                        height: float = 2.0,
                        seed: int = 0,
                        noise_scale: float = 1.0,
                        # Расширенные параметры
                        name: str = "Landscape",
                        subdivisions_x: int = None,
                        subdivisions_y: int = None,
                        mesh_size_x: float = None,
                        mesh_size_y: float = None,
                        noise_type: str = "rocks_noise",
                        noise_basis: str = "BLENDER",
                        offset_x: float = -1.05,
                        offset_y: float = 0.0,
                        size_x: float = 1.45,
                        size_y: float = 2.23,
                        depth: int = 8,
                        distortion: float = 1.39,
                        hard_noise: str = "0",
                        height_offset: float = 0.0,
                        maximum: float = 10000.0,
                        minimum: float = -10000.0,
                        edge_falloff: str = "3",
                        edge_level: float = -0.12,
                        falloff_x: float = 3.70,
                        falloff_y: float = 4.00,
                        strata_type: str = "0",
                        output_format: str = "ply",
                        # Повторение текстуры
                        texture_repeat_x: float = 1.35,
                        texture_repeat_y: float = 3.2
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Генерирует ландшафт через Blender и возвращает:
        vertices (N,3), triangles (M,3), normals (N,3), uvs (N,2)
        """
        if subdivisions_x is None:
            subdivisions_x = subdivisions
        if subdivisions_y is None:
            subdivisions_y = subdivisions
        if mesh_size_x is None:
            mesh_size_x = size
        if mesh_size_y is None:
            mesh_size_y = size

        payload = {
            "name": name,
            "size": size,
            "subdivisions": subdivisions,
            "subdivisions_x": subdivisions_x,
            "subdivisions_y": subdivisions_y,
            "mesh_size_x": mesh_size_x,
            "mesh_size_y": mesh_size_y,
            "height": height,
            "seed": seed,
            "noise_scale": noise_scale,
            "noise_type": noise_type,
            "noise_basis": noise_basis,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "size_x": size_x,
            "size_y": size_y,
            "depth": depth,
            "distortion": distortion,
            "hard_noise": hard_noise,
            "height_offset": height_offset,
            "maximum": maximum,
            "minimum": minimum,
            "edge_falloff": edge_falloff,
            "edge_level": edge_level,
            "falloff_x": falloff_x,
            "falloff_y": falloff_y,
            "strata_type": strata_type,
            "output_format": output_format,
            "texture_repeat_x": texture_repeat_x,
            "texture_repeat_y": texture_repeat_y
        }
        response = self._post("generate_landscape", payload)
        if response.get("status") != "success":
            raise RuntimeError(f"Landscape generation failed: {response.get('error', 'Unknown')}")

        verts_b64 = response["vertices"]
        tris_b64  = response["triangles"]
        norms_b64 = response["normals"]
        uvs_b64   = response["uvs"]

        vertices  = np.frombuffer(base64.b64decode(verts_b64), dtype=np.float32).reshape(-1, 3)
        triangles = np.frombuffer(base64.b64decode(tris_b64),  dtype=np.uint32).reshape(-1, 3)
        normals   = np.frombuffer(base64.b64decode(norms_b64), dtype=np.float32).reshape(-1, 3)
        uvs       = np.frombuffer(base64.b64decode(uvs_b64),   dtype=np.float32).reshape(-1, 2)
        return vertices, triangles, normals, uvs

    # ----------------- остальные методы (boolean, конфиги и т.д.) без изменений -----------------
    def send_boolean_intersection(self,
                                mesh1_vertices: np.ndarray,
                                mesh1_triangles: np.ndarray,
                                mesh2_vertices: np.ndarray,
                                mesh2_triangles: np.ndarray,
                                return_volume_only: bool = False) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        v1 = np.asarray(mesh1_vertices, dtype=np.float32)
        t1 = np.asarray(mesh1_triangles, dtype=np.uint32)
        v2 = np.asarray(mesh2_vertices, dtype=np.float32)
        t2 = np.asarray(mesh2_triangles, dtype=np.uint32)
        payload = {
            "type": "boolean_intersection",
            "mesh1_vertices": base64.b64encode(v1.tobytes()).decode('ascii'),
            "mesh1_triangles": base64.b64encode(t1.tobytes()).decode('ascii'),
            "mesh2_vertices": base64.b64encode(v2.tobytes()).decode('ascii'),
            "mesh2_triangles": base64.b64encode(t2.tobytes()).decode('ascii'),
            "return_volume_only": "true" if return_volume_only else "false"
        }
        response = self._post("boolean_intersection", payload)
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

    def send_boolean_request(self,
                             mesh1_vertices: np.ndarray,
                             mesh1_triangles: np.ndarray,
                             mesh2_vertices: np.ndarray,
                             mesh2_triangles: np.ndarray,
                             return_volume_only: bool = False) -> Union[float, Tuple[np.ndarray, np.ndarray]]:
        v1 = np.asarray(mesh1_vertices, dtype=np.float32)
        t1 = np.asarray(mesh1_triangles, dtype=np.uint32)
        v2 = np.asarray(mesh2_vertices, dtype=np.float32)
        t2 = np.asarray(mesh2_triangles, dtype=np.uint32)
        payload = {
            "type": "boolean_difference",
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

    def get_models_config(self) -> dict:
        response = self._post("get_models_config", {})
        if response.get("status") != "success":
            raise RuntimeError(f"Ошибка получения конфига моделей: {response.get('error', 'Неизвестная ошибка')}")
        return response.get("config", {})

    def download_model_file(self, set_name: str, file_type: str, local_path: str) -> None:
        url = f"{self.base_url}/download_model_file?set={urllib.parse.quote(set_name)}&type={urllib.parse.quote(file_type)}"
        try:
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            raise RuntimeError(f"Ошибка скачивания файла {file_type} для набора {set_name}: {e}")

    def get_texture_list(self, textures_dir: str) -> List[str]:
        params = {"dir": textures_dir}
        resp = self._get("list_textures", params)
        data = resp.json()
        if data.get("status") != "success":
            raise RuntimeError(f"Ошибка получения списка текстур: {data.get('error', 'Неизвестная ошибка')}")
        return data.get("files", [])

    def download_texture_file(self, textures_dir: str, filename: str, local_path: str) -> None:
        params = {"dir": textures_dir, "file": filename}
        url = f"{self.base_url}/download_texture"
        try:
            resp = requests.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(resp.content)
        except Exception as e:
            raise RuntimeError(f"Ошибка скачивания текстуры {filename} из {textures_dir}: {e}")