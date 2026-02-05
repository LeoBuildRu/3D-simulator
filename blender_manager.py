# blender_manager.py
"""
BlenderManager - persistent Blender server runner and boolean wrapper.

Compatibility notes:
- Embedded Blender server script is written to support Blender 2.7x (older Python)
  and Blender 2.8+/3.x (newer API). The script intentionally avoids f-strings
  and other Python 3.6+ syntax so it runs on Python shipped with Blender 2.77a.
- This manager starts Blender once (background), sends boolean requests over TCP,
  and returns a single trimesh.Trimesh for boolean_difference calls.
"""

import os
import sys
import json
import socket
import shutil
import tempfile
import subprocess
import time
from typing import List, Union, Optional

import trimesh

DEFAULT_PORT = 50510
DEFAULT_HOST = "127.0.0.1"

# -------------------------------------------------------------------------
# Embedded Blender server script (compatible with Blender 2.7x Python)
# -------------------------------------------------------------------------
_BLENDER_SERVER_PY = r'''
# blender_boolean_server_compat.py
# Runs inside Blender (--background --python thisfile.py -- --port N)
# Compatible with Blender 2.7x and 2.8+/3.x Python versions (avoids f-strings).

import sys
import os
import json
import socket
import traceback
import bpy

HOST = "127.0.0.1"
DEFAULT_PORT = 50510

IS_28 = bpy.app.version >= (2, 80, 0)

def debug_print(*args, **kwargs):
    # Print and flush for manager to detect readiness/diagnostics
    try:
        sys.stdout.write(" ".join(str(a) for a in args) + ("\n" if kwargs.get("end", "\n") == "\n" else ""))
        sys.stdout.flush()
    except Exception:
        try:
            print(*args, **kwargs)
            sys.stdout.flush()
        except Exception:
            pass

def select_obj(obj, state=True):
    if obj is None:
        return
    if hasattr(obj, "select_set"):
        try:
            obj.select_set(state)
            return
        except Exception:
            pass
    try:
        obj.select = state
    except Exception:
        pass

def active_object_set(obj):
    try:
        if IS_28:
            bpy.context.view_layer.objects.active = obj
        else:
            bpy.context.scene.objects.active = obj
    except Exception:
        pass

def import_obj_return_new(filepath):
    before = set(bpy.data.objects)
    # use operator for import (works across versions)
    bpy.ops.import_scene.obj(filepath=filepath)
    after = set(bpy.data.objects)
    new = list(after - before)
    return new

def export_obj_single(obj, out_path):
    # Deselect everything
    try:
        for o in list(bpy.data.objects):
            select_obj(o, False)
    except Exception:
        pass

    # Link to scene/collection if needed
    try:
        if IS_28:
            if obj.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(obj)
        else:
            if obj.name not in bpy.context.scene.objects:
                bpy.context.scene.objects.link(obj)
    except Exception:
        pass

    select_obj(obj, True)
    active_object_set(obj)

    # Export selected object
    try:
        bpy.ops.export_scene.obj(filepath=out_path, use_selection=True, use_materials=False, use_triangles=True)
    except Exception:
        # fallback: try export with defaults (may export entire scene)
        try:
            bpy.ops.export_scene.obj(filepath=out_path, use_materials=False, use_triangles=True)
        except Exception:
            pass

    select_obj(obj, False)

def ensure_linked(obj):
    try:
        if IS_28:
            if obj.name not in bpy.context.scene.collection.objects:
                bpy.context.scene.collection.objects.link(obj)
        else:
            if obj.name not in bpy.context.scene.objects:
                bpy.context.scene.objects.link(obj)
    except Exception:
        pass

def apply_boolean_difference_and_bake(base_obj, cutter_objs):
    if base_obj is None:
        raise RuntimeError("base_obj is None")

    # Attach boolean modifiers
    for i in range(len(cutter_objs)):
        cutter = cutter_objs[i]
        if cutter is None:
            continue
        try:
            mod = base_obj.modifiers.new("bool_{}".format(i), 'BOOLEAN')
            mod.operation = 'DIFFERENCE'
            mod.object = cutter
        except Exception:
            # ignore and continue
            pass

    mesh_temp = None
    mesh_copy = None
    try:
        if IS_28:
            deps = bpy.context.evaluated_depsgraph_get()
            base_eval = base_obj.evaluated_get(deps)
            # to_mesh returns a temporary evaluated mesh
            mesh_temp = base_eval.to_mesh()
            if mesh_temp is None:
                raise RuntimeError("evaluated_get().to_mesh() returned None")
            # copy to persistent datablock
            mesh_copy = mesh_temp.copy()
            mesh_copy.name = "boolean_result_mesh"
            # clear temporary if API provides
            try:
                base_eval.to_mesh_clear()
            except Exception:
                try:
                    if mesh_temp.name in bpy.data.meshes:
                        bpy.data.meshes.remove(mesh_temp)
                except Exception:
                    pass
        else:
            # Blender 2.7x path
            mesh_temp = base_obj.to_mesh(bpy.context.scene, True, 'PREVIEW')
            if mesh_temp is None:
                raise RuntimeError("to_mesh(...) returned None (2.7)")
            mesh_copy = mesh_temp.copy()
            mesh_copy.name = "boolean_result_mesh"
            try:
                if mesh_temp.name in bpy.data.meshes:
                    bpy.data.meshes.remove(mesh_temp)
            except Exception:
                pass

        # create new object with persistent mesh copy and link it
        result_obj = bpy.data.objects.new("boolean_result", mesh_copy)
        ensure_linked(result_obj)
        return result_obj

    except Exception:
        # try to cleanup mesh_temp if present
        try:
            if mesh_temp is not None and hasattr(mesh_temp, "name") and mesh_temp.name in bpy.data.meshes:
                bpy.data.meshes.remove(mesh_temp)
        except Exception:
            pass
        raise

def safe_remove_object(obj):
    try:
        bpy.data.objects.remove(obj, do_unlink=True)
    except Exception:
        try:
            bpy.data.objects.remove(obj)
        except Exception:
            pass

def run_boolean_request(input_paths, out_path):
    created = []
    try:
        # import base
        base_objs = import_obj_return_new(input_paths[0])
        if not base_objs:
            return False, "Base import produced no objects"
        base_obj = base_objs[0]
        created.extend(base_objs)

        # import cutters
        cutters = []
        for p in input_paths[1:]:
            c_objs = import_obj_return_new(p)
            if not c_objs:
                continue
            cutters.append(c_objs[0])
            created.extend(c_objs)

        # ensure linked
        ensure_linked(base_obj)
        for c in cutters:
            ensure_linked(c)

        # apply booleans and bake
        result_obj = apply_boolean_difference_and_bake(base_obj, cutters)
        created.append(result_obj)

        # export result
        export_obj_single(result_obj, out_path)

        # cleanup created objects
        for o in created:
            try:
                safe_remove_object(o)
            except Exception:
                pass
                
        # HARD PURGE (prevents 2.7 memory leak / freezes)
        try:
            for m in list(bpy.data.meshes):
                if m.users == 0:
                    bpy.data.meshes.remove(m)

            for o in list(bpy.data.objects):
                if o.users == 0:
                    bpy.data.objects.remove(o)

            # safest: reset whole scene
            bpy.ops.wm.read_factory_settings(use_empty=True)
        except Exception:
            pass

        return True, "ok"

    except Exception as exc:
        tb = traceback.format_exc()
        # best-effort cleanup omitted
        return False, "{}\n\nTRACEBACK:\n{}".format(str(exc), tb)

def serve(port):
    debug_print("BLENDER_BOOLEAN_SERVER_READY")
    try:
        sys.stdout.flush()
    except Exception:
        pass

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, port))
    s.listen(1)
    while True:
        conn, addr = s.accept()
        try:
            data = b''
            conn.settimeout(5.0)
            while True:
                try:
                    chunk = conn.recv(65536)
                except socket.timeout:
                    break
                if not chunk:
                    break
                data += chunk
                try:
                    msg = json.loads(data.decode('utf8'))
                    break
                except Exception:
                    continue

            if not data:
                resp = {'ok': False, 'error': 'empty request'}
                conn.send(json.dumps(resp).encode('utf8'))
                conn.close()
                continue

            try:
                msg = json.loads(data.decode('utf8'))
            except Exception as e:
                resp = {'ok': False, 'error': 'invalid JSON: {}'.format(e)}
                conn.send(json.dumps(resp).encode('utf8'))
                conn.close()
                continue

            inputs = msg.get('inputs') or msg.get('meshes') or []
            output = msg.get('output') or msg.get('out')
            op = msg.get('op', 'difference')

            if not output:
                resp = {'ok': False, 'error': 'no output path provided'}
                conn.send(json.dumps(resp).encode('utf8'))
                conn.close()
                continue

            if op != 'difference':
                resp = {'ok': False, 'error': 'unsupported op: {}'.format(op)}
                conn.send(json.dumps(resp).encode('utf8'))
                conn.close()
                continue

            ok, info = run_boolean_request(inputs, output)
            resp = {'ok': ok, 'info': info} if ok else {'ok': False, 'error': info}
            conn.send(json.dumps(resp).encode('utf8'))
            conn.close()
        except Exception as e:
            try:
                conn.send(json.dumps({'ok': False, 'error': str(e), 'trace': traceback.format_exc()}).encode('utf8'))
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

if __name__ == "__main__":
    argv = sys.argv
    port = DEFAULT_PORT
    if '--' in argv:
        try:
            idx = argv.index('--')
            args = argv[idx+1:]
            if '--port' in args:
                pidx = args.index('--port')
                port = int(args[pidx+1])
        except Exception:
            pass
    try:
        serve(port)
    except Exception:
        debug_print("Server exception:\n{}".format(traceback.format_exc()))
'''

# -------------------------------------------------------------------------
# BlenderManager implementation
# -------------------------------------------------------------------------
class BlenderManager:
    def __init__(self,
                 blender_executable: Optional[str] = None,
                 server_port: int = DEFAULT_PORT,
                 server_script_path: Optional[str] = None,
                 verbose: bool = True):
        """
        blender_executable: optional full path to blender binary. If None, will try to locate in PATH.
        server_port: port for Blender server to listen on.
        server_script_path: optional path to custom server script to run inside Blender. If None,
                            a temporary server script is written from the builtin template.
        """
        self.blender_executable = blender_executable or self._find_blender_executable()
        self.server_port = server_port
        self.server_script_path = server_script_path
        self.server_process = None
        self._server_script_is_temp = False
        self.blender_python = None
        self.verbose = verbose
        self.calls = 0

        if not self.blender_executable:
            raise RuntimeError("Blender executable not found. Pass blender_executable explicitly or ensure 'blender' is in PATH.")

    # -------------------------
    # Setup helpers
    # -------------------------
    def _log(self, *args, **kwargs):
        if self.verbose:
            print("[BlenderManager]", *args, **kwargs)

    def _find_blender_executable(self) -> Optional[str]:
        env = os.environ.get('BLENDER_PATH')
        if env and os.path.isfile(env):
            return env
        blender = shutil.which("blender")
        if blender:
            return blender
        possible = [
            r"C:\Program Files\Blender Foundation\Blender\blender.exe",
            r"C:\Program Files\Blender Foundation\blender\blender.exe",
            "/usr/bin/blender",
            "/usr/local/bin/blender"
        ]
        for p in possible:
            if os.path.exists(p):
                return p
        return None

    def _query_blender_python(self) -> str:
        if self.blender_python:
            return self.blender_python

        cmd = [self.blender_executable, '--background', '--python-expr',
               'import sys, json; print(json.dumps(sys.executable))']
        self._log("Querying Blender's Python executable with:", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=20)
            out = proc.stdout.strip().splitlines()[-1]
            try:
                python_path = json.loads(out)
            except Exception:
                python_path = out.strip()
            self.blender_python = python_path
            self._log("Blender python is:", python_path)
            return python_path
        except subprocess.CalledProcessError as e:
            self._log("Failed to query blender's python:", e, e.stdout, e.stderr)
            raise
        except Exception as e:
            self._log("Error querying blender python:", e)
            raise

    def ensure_blender_python_packages(self, packages: List[str]):
        """
        Ensure given pip-installable packages are installed in Blender's bundled Python.
        Example: ensure_blender_python_packages(['numpy', 'trimesh'])
        """
        python_exe = self._query_blender_python()
        if not python_exe or not os.path.exists(python_exe):
            raise RuntimeError("Blender python executable not found")

        for pkg in packages:
            cmd_check = [python_exe, '-c', 'import importlib, sys\ntry:\n importlib.import_module("{0}")\n print("OK")\nexcept Exception as e:\n print("MISSING")'.format(pkg)]
            try:
                r = subprocess.run(cmd_check, capture_output=True, text=True, timeout=30)
                out = r.stdout.strip()
                if out.endswith("OK"):
                    self._log("Package '{0}' already present in Blender Python.".format(pkg))
                    continue
            except Exception:
                pass
            self._log("Installing '{0}' into Blender Python: {1} -m pip install {0}".format(pkg, python_exe))
            cmd_install = [python_exe, '-m', 'pip', 'install', pkg]
            proc = subprocess.run(cmd_install, capture_output=True, text=True, timeout=300)
            if proc.returncode != 0:
                self._log("pip install failed:", proc.stdout, proc.stderr)
                raise RuntimeError("Failed to install {0} into Blender Python: {1}".format(pkg, proc.stderr))
            self._log("Installed '{0}' into Blender Python.".format(pkg))

    # -------------------------
    # Server lifecycle
    # -------------------------
    def start_server(self, server_script_path: Optional[str] = None, force_restart: bool = False, wait_ready: bool = True, timeout: int = 20):
        """
        Start Blender in background running the server script.
        If server_script_path is None, write a temp script from the embedded template.
        """
        if self.server_process and not force_restart:
            if self.server_process.poll() is None:
                self._log("Server already running (pid={0})".format(self.server_process.pid))
                return
            else:
                self.server_process = None

        if server_script_path is None:
            fd, path = tempfile.mkstemp(suffix="_blender_boolean_server.py", text=True)
            with os.fdopen(fd, "w", encoding="utf8") as f:
                f.write(_BLENDER_SERVER_PY)
            server_script_path = path
            self._server_script_is_temp = True
            self._log("Wrote embedded blender server script to", path)
        else:
            self._server_script_is_temp = False

        self.server_script_path = server_script_path

        cmd = [self.blender_executable, '--background', '--python', server_script_path, '--', '--port', str(self.server_port)]
        self._log("Starting Blender server with command:", " ".join(cmd))
        self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True)

        if wait_ready:
            start = time.time()
            ready = False
            out_lines = []
            while time.time() - start < timeout:
                if self.server_process.poll() is not None:
                    stderr = ""
                    stdout = ""
                    try:
                        stderr = self.server_process.stderr.read() if self.server_process.stderr else ""
                        stdout = self.server_process.stdout.read() if self.server_process.stdout else ""
                    except Exception:
                        pass
                    raise RuntimeError("Blender server process terminated unexpectedly.\nstdout:\n{0}\nstderr:\n{1}".format(stdout, stderr))
                line = self.server_process.stdout.readline()
                if not line:
                    time.sleep(0.05)
                    continue
                out_lines.append(line.strip())
                if "BLENDER_BOOLEAN_SERVER_READY" in line:
                    ready = True
                    break
            if not ready:
                stderr_all = ""
                try:
                    stderr_all = self.server_process.stderr.read() if self.server_process.stderr else ""
                except Exception:
                    pass
                self._log("Server did not report ready within timeout. stdout (recent):", out_lines[-10:])
                self._log("stderr:", stderr_all)
                raise RuntimeError("Blender server did not start in time. Check blender executable and server script.")
        self._log("Blender server started, pid={0}".format(self.server_process.pid))

    def stop_server(self, kill: bool = False):
        """Stop the Blender server if running."""
        if not self.server_process:
            return
        try:
            if kill:
                self.server_process.kill()
            else:
                self.server_process.terminate()
        except Exception:
            try:
                self.server_process.kill()
            except Exception:
                pass
        self.server_process.wait(timeout=5)
        self._log("Blender server stopped.")
        self.server_process = None
        if self._server_script_is_temp and self.server_script_path:
            try:
                os.remove(self.server_script_path)
            except Exception:
                pass
            self._server_script_is_temp = False
            self.server_script_path = None

    # -------------------------
    # Communication
    # -------------------------
    def _call_server_boolean(self, input_paths: List[str], timeout: int = 60) -> str:
        """Send a boolean request to Blender server. Returns path to output obj on success."""
        if not self.server_process or self.server_process.poll() is not None:
            raise RuntimeError("Blender server is not running. Call start_server() first.")

        out_fd, out_path = tempfile.mkstemp(suffix="_blender_boolean_out.obj")
        os.close(out_fd)
        os.remove(out_path)  # server will create

        req = {
            "inputs": input_paths,
            "output": out_path,
            "op": "difference"
        }
        data = json.dumps(req).encode('utf8')

        with socket.create_connection((DEFAULT_HOST, self.server_port), timeout=timeout) as s:
            s.sendall(data)
            resp_bytes = b''
            while True:
                s.settimeout(timeout)
                try:
                    chunk = s.recv(10000) # 10 seconds
                except socket.timeout:
                    raise RuntimeError(
                        "Blender server hung during boolean (likely Blender 2.7 boolean stall). "
                        "Consider restarting server or reducing mesh complexity."
                    )
                if not chunk:
                    break
                resp_bytes += chunk
            if not resp_bytes:
                raise RuntimeError("No response from Blender server")
            try:
                resp = json.loads(resp_bytes.decode('utf8'))
            except Exception:
                raise RuntimeError("Invalid JSON response from Blender server: {0!r}".format(resp_bytes))
            if not resp.get('ok', False):
                msg = resp.get('error') or resp.get('info') or 'unknown error'
                raise RuntimeError("Blender boolean failed: {0}".format(msg))
            return out_path

    # -------------------------
    # Public boolean API
    # -------------------------
    def boolean_difference(self, inputs: List[Union[str, trimesh.Trimesh]], engine: str = 'blender', timeout: int = 60) -> trimesh.Trimesh:
        """
        Perform boolean difference using Blender server.

        inputs: list where first is minuend (A), subsequent entries are subtrahends (B1, B2, ...).
                Each element can be:
                  - path to a mesh file (OBJ/PLY/glTF supported)
                  - trimesh.Trimesh instance (will be exported to temporary OBJ)
        engine: must be 'blender' for this manager (keeps API compatibility)
        timeout: seconds to wait for server response

        Returns: trimesh.Trimesh loaded from blender's result OBJ (always a single mesh).
        """
        if engine != 'blender':
            raise ValueError("This BlenderManager only supports engine='blender'")

        temp_files = []
        try:
            input_paths = []
            for idx, item in enumerate(inputs):
                if isinstance(item, str):
                    if not os.path.exists(item):
                        raise FileNotFoundError("Input mesh path not found: {0}".format(item))
                    input_paths.append(item)
                elif isinstance(item, trimesh.Trimesh):
                    fd, path = tempfile.mkstemp(suffix="_bm_in_{0}.obj".format(idx))
                    os.close(fd)
                    temp_files.append(path)
                    item.export(path, file_type='obj')
                    input_paths.append(path)
                else:
                    raise TypeError("Unsupported input type: must be file path or trimesh.Trimesh")

            try:
                out_path = self._call_server_boolean(input_paths, timeout=timeout)
                self.calls += 1
            except RuntimeError:
                self._log(f"Restarting Blender server after stall after {self.calls} calls...")
                self.stop_server(kill=True)
                self.calls = 0
                self.start_server()
                out_path = self._call_server_boolean(input_paths, timeout=timeout)

            loaded = None
            try:
                loaded = trimesh.load(out_path, force='scene')
            except Exception:
                loaded = trimesh.load(out_path, force='mesh')

            result_mesh = None

            if isinstance(loaded, trimesh.Scene):
                # try scene dump(concatenate=True) if available
                try:
                    if hasattr(loaded, 'dump'):
                        try:
                            mesh_try = loaded.dump(concatenate=True)
                            if isinstance(mesh_try, trimesh.Trimesh):
                                result_mesh = mesh_try
                        except Exception:
                            pass
                except Exception:
                    pass

                if result_mesh is None:
                    geoms = []
                    for name, geom in loaded.geometry.items():
                        if isinstance(geom, trimesh.Trimesh):
                            geoms.append(geom.copy())
                    if not geoms:
                        raise RuntimeError("Blender result contains no geometry: {0}".format(out_path))
                    if len(geoms) == 1:
                        result_mesh = geoms[0]
                    else:
                        result_mesh = trimesh.util.concatenate(geoms)

            elif isinstance(loaded, trimesh.Trimesh):
                result_mesh = loaded

            elif isinstance(loaded, (list, tuple)):
                mesh_list = []
                for item in loaded:
                    if isinstance(item, trimesh.Trimesh):
                        mesh_list.append(item)
                if not mesh_list:
                    raise RuntimeError("Blender result list contained no valid meshes: {0}".format(out_path))
                result_mesh = trimesh.util.concatenate(mesh_list)

            else:
                raise RuntimeError("Unexpected result type from trimesh.load: {0}; path={1}".format(type(loaded), out_path))

            if result_mesh is None:
                raise RuntimeError("Failed to assemble result mesh from Blender output: {0}".format(out_path))

            # remove temporary output
            try:
                os.remove(out_path)
            except Exception:
                pass

            return result_mesh

        finally:
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception:
                    pass

    # alias
    difference = boolean_difference

# -----------------------------------------------------------------------------
# Quick demo
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    bm = BlenderManager(verbose=True)
    print("Blender executable: {0}".format(bm.blender_executable))
    try:
        bm.ensure_blender_python_packages(['numpy'])
    except Exception as e:
        print("ensure packages warning:", e)
    bm.start_server(wait_ready=True)
    try:
        a = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        b = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
        b.apply_translation((0.2, 0.0, 0.0))
        res = bm.boolean_difference([a, b], timeout=30)
        print("Result loaded. vertices: {0}, faces: {1}".format(len(res.vertices), len(res.faces)))
    finally:
        bm.stop_server(kill=True)
