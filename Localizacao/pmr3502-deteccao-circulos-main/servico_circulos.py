#!/usr/bin/env python3
'''
	Author: Thiago Martins.
'''
import cv2
import logging
import os
import aiohttp
import asyncio
from asyncio import Queue
from aiohttp import web, MultipartWriter
import sys
import tempfile
import subprocess
import argparse
import numpy as np
import time
import contextlib
from picamera2 import Picamera2


# Arquivo esperado dentro da pasta Localizacao.
LOCALIZACAO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CALIB_PATH = os.path.join(LOCALIZACAO_DIR, "camera_calibration_results.npz")

# Parâmetros usados no final de odometria_visual.ipynb para o remapeamento
# completo no solo.
PIXELS_POR_MM = 2.0
XMAX_MM = 400.0
YMAX_MM = 0.8 * XMAX_MM
CAMERA_SIZE = (1296, 972)


def carrega_calibracao():
    if not os.path.exists(CALIB_PATH):
        raise FileNotFoundError(
            "Coloque camera_calibration_results.npz em: " + CALIB_PATH)

    calib = np.load(CALIB_PATH)
    return calib["mtx"], calib["dist"], calib["s"]


def cria_mascara(map_x, map_y, largura, altura):
    # Mascara da area visivel: pixels da imagem reprojetada cujo mapa aponta
    # para dentro dos limites do quadro original.
    valida = (
        (map_x >= 0) & (map_x <= largura - 1) &
        (map_y >= 0) & (map_y <= altura - 1)
    )
    mascara = np.where(valida, 255, 0).astype(np.uint8)
    mascara = cv2.erode(mascara, np.ones((9, 9), np.uint8), iterations=2)
    return mascara


def cria_matriz_reprojecao():
    Q = np.float32([
        [PIXELS_POR_MM, 0, 0],
        [0, PIXELS_POR_MM, PIXELS_POR_MM * YMAX_MM],
        [0, 0, 1],
    ])
    dim = (
        int(PIXELS_POR_MM * XMAX_MM),
        int(PIXELS_POR_MM * 2 * YMAX_MM),
    )
    return Q, dim


# Estima o momento do boot
def getboottime():
    times = []
    for i in range(40):
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
        times.append(t2-t1)
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)
        t2 = time.clock_gettime(time.CLOCK_REALTIME)
        times.append(t2-t1)

    times.sort()
    tot = 0.0
    for i in range(15,25):
        tot += times[i]
    return tot/10



class Detector():
    def __init__(self, min_radius, max_radius, camera):
        mtx, dist, s = carrega_calibracao()
        Q, dim = cria_matriz_reprojecao()

        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            mtx, dist, s, Q, dim, cv2.CV_32FC1)
        self._mask = cria_mascara(
            self._map_x, self._map_y, CAMERA_SIZE[0], CAMERA_SIZE[1])
        self._scale = PIXELS_POR_MM
        self._itrans_matrix = np.linalg.inv(Q)

        # Parâmetros minRadius e maxRadius
        self._min_radius = np.int32(np.floor(min_radius*self._scale))
        self._max_radius = np.int32(np.ceil(max_radius*self._scale))
        self._boot_time = getboottime()*1000000000
        self._prev_frame_ts = time.time()
        self._vid = camera

    def detecta(self):
        req = None
        try:
            req = self._vid.capture_request()
            frame = req.make_array("main")
            data = req.get_metadata()
        finally:
            if req is not None:
                req.release()

        ts = int((data['SensorTimestamp'])+self._boot_time)
        # Imagem em tons de cinza
        cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        circles = None

        imagem = cv2.remap(cinza, self._map_x, self._map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        imagem = cv2.bitwise_and(imagem, imagem, mask=self._mask)
        imagem = cv2.medianBlur(imagem, 3)
        circles = cv2.HoughCircles(
            imagem,
            method=cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=max(20, 2 * int(self._min_radius)),
            param1=50,
            param2=15,
            minRadius=int(self._min_radius),
            maxRadius=int(self._max_radius)
        )

        coordenadas = []
        if circles is not None:
            for c in circles[0, :]:
                coordnumpy = self._itrans_matrix.dot(np.float32([c[0], c[1], 1.0]))
                coordenadas.append([float(coordnumpy[0]), float(coordnumpy[1])])
            ncircles = len(circles[0, :])
        else:
            ncircles = 0
        t = time.time()
        print("n: " + str(ncircles) + " FPS: " + str(1/(t-self._prev_frame_ts)) + " lag: " + str(time.time() - ts/1000000000), end="\r")
        self._prev_frame_ts = t
        return coordenadas, ts

# Cria detector de círculos.
#   Este gerador é decorado com contextmanager para garantir
#   que o estado da câmera seja resetado na saída de escopo
@contextlib.contextmanager
def cria_detector_circulos(min_radius, max_radius):
    # Atribui nível "erro" para libcamera e picamera
    Picamera2.set_logging(Picamera2.ERROR)
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    try:
        with Picamera2(tuning=os.environ.get('LIBCAMERA_RPI_TUNING_FILE', None)) as camera:
            camera.configure(camera.create_still_configuration(main={"size": CAMERA_SIZE}))
            camera.start()
            yield Detector(min_radius, max_radius, camera)
            camera.stop()
    finally:
        pass

class ServicoDetectorCirculos():

    def __init__(self, app, endereco_servidor, porta_servidor, detector):
        self._app = app
        self._endereco_servidor = endereco_servidor
        self._porta_servidor = porta_servidor
        self._app['app_object'] = self
        # Tarefas de inicializacao e encerramento
        self._app.on_startup.append(self._inicializa_tarefas)
        self._app.on_cleanup.append(self._encerra_tarefas)
        self._app.router.add_routes([web.get('/wsctrl', self._websocket_handler)])
        self._app.router.add_routes([web.get('/', self._pagina)])
        self._keep_alive = True
        self._worker_task = None
        self._connections = set()
        self._detector = detector

    def run(self):
        web.run_app(self._app, host=self._endereco_servidor, port=self._porta_servidor,  shutdown_timeout=0.2)

    async def _pagina(self, request):
        return web.FileResponse('./static/showcircles.html')

    async def _inicializa_tarefas(self, app):
        self._worker_task = asyncio.create_task(ServicoDetectorCirculos._worker(self._app))

    async def _encerra_tarefas(self, app):
        self._keep_alive = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            await self._worker_task
            self._worker_task = None

    # Responde a uma conexão web socket
    async def _websocket_handler(self, request):
        messages = Queue(1)
        self._connections.add(messages)
        ws = web.WebSocketResponse(receive_timeout=0)
        await ws.prepare(request)
        try:
            while self._keep_alive and not ws.closed:
                try:
                    from_client = await ws.receive(timeout = 0)
                    if from_client.type==web.WSMsgType.CLOSE:
                        break
                    elif from_client.type==web.WSMsgType.TEXT:
                        pass
                except asyncio.TimeoutError as e:
                    pass

                msg = await messages.get()
                messages.task_done()
                await ws.send_json(msg)
        except Exception as e:
            print("ERROR")
            print(e.__class__.__qualname__)
        self._connections.remove(messages)


    # Detecta os círculos
    async def _worker(app):
        self = app['app_object']
        while self._keep_alive:
            await asyncio.sleep(0)
            coordenadas, timestamp = self._detector.detecta()
            dados = {"timestamp": timestamp, "coordenadas" : coordenadas }
            for connection in self._connections:
                if connection.full():   # Remove dados de clientes backlogged
                    connection.get_nowait()
                connection.put_nowait(dados)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="Endereço externo do servidor")
    parser.add_argument('-p', help="Porta do servidor", default="8086")
    parser.add_argument('-a', help="Raio máximo do círculo", default="18")
    parser.add_argument('-i', help="Raio mínimo do círculo", default="9")

    args = parser.parse_args()
    endereco_servidor = args.e
    porta_servico = args.p
    raio_minimo = args.i
    raio_maximo = args.a

    if endereco_servidor == None:
        endereco_servidor = "0.0.0.0"

    print("Endereço do servidor: " + endereco_servidor)
    print("Porta do servidor: " + porta_servico)

    with cria_detector_circulos(int(raio_minimo), int(raio_maximo)) as detector:
        serviceObj = ServicoDetectorCirculos(web.Application(), endereco_servidor, int(porta_servico), detector)
        serviceObj.run()

    return 0

if __name__ == '__main__':
    sys.exit(main())
