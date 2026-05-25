#!/usr/bin/env python3
'''
	Author: Thiago Martins.
	"Inspirado" pelo servidor mjpeg feito por Igor Maculan
'''
import cv2
import logging
import os
import aiohttp
import asyncio
from aiohttp import web, MultipartWriter
import sys
import tempfile
import subprocess
import getopt
from threading import Condition
import io
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import os

# Página principal
#   Só uma tag <img> com o endereço do serviço mjpeg
#   e um botão que envia um comando websocket
pag_template = r"""<!DOCTYPE HTML>
<html>
	<head>
		<title>Captura de video</title>
		<meta charset="utf-8">
	</head>
	<body>
        <center>
            <div><img src="/image" /></div>
            <div><button id="botao_capturar" type="button">Capturar Imagem</button></div>
        </center>
		<script type="text/javascript">
var wsUri = (window.location.protocol=='https:'&&'wss://'||'ws://')+window.location.host+"/wsctrl";
var control_link = new WebSocket(wsUri);
document.getElementById("botao_capturar").onclick = function() {{envia_comando_capturar()}};
function envia_comando_capturar(){{
    control_link.send("capturar");
}}
		</script>
	</body>
</html>
"""

class CallbackWriter(io.BufferedIOBase):

    def __init__(self, callback):
        self._writecallback = callback

    def write(self, buff):
        self._writecallback(buff)

class StreamingAndCaptureServer():

    def __init__(self, capture_path):
        self._app = web.Application()
        self._capture_path = capture_path
        self._last_frame = None
        self._connections = set()
        self._camera = None
        self._stop_camera_task = None
        self._app.on_startup.append(self.inicializa_tarefas)
        self._app.on_cleanup.append(self.encerra_tarefas)
        self._camera_running = False
        # Página raiz
        self._app['root_pag'] = pag_template
        self._app.router.add_route("GET", "/", self.root_handler)
        # Fluxo mjpeg
        self._app.add_routes([web.get('/image', self.mjpeg_request)])
        # Comando para gravar
        self._app.router.add_routes([web.get('/wsctrl', self.websocket_handler)])

    def run(self, endereco_servidor, porta_servico):
        self._camera = self._cria_camera()
        web.run_app(self._app, host=endereco_servidor, port=int(porta_servico), shutdown_timeout=0.2)

    def _cria_camera(self):
        picam2 = Picamera2(tuning=os.environ.get('LIBCAMERA_RPI_TUNING_FILE', None))
        picam2.configure(picam2.create_video_configuration(main={"size": (1296, 972)}))
        picam2.start(show_preview = False)
        return picam2

    def process_new_frame(self, frame):
        self._last_frame = frame
        for queue in self._connections:
            try:
                queue.put_nowait(frame)
            except asyncio.QueueFull: # Client backlogged, skip frame
                pass

    def _start_camera(self):
        if not self._camera_running:
            self._camera.start_recording(JpegEncoder(), FileOutput(self._output))
            self._camera_running = True

    def _stop_camera(self):
        if self._camera_running:
            self._camera_running = False
            self._camera.stop_recording()
        if self._stop_camera_task:
            self._stop_camera_task.cancel()
            self._stop_camera_task = None

    def _create_new_stream_client(self):
        if len(self._connections)==0:   # Primeira conexão
            # Se há interrupção de câmera pendente, cancela
            if self._stop_camera_task:
               self._stop_camera_task.cancel()
            # Caso contrário, inicia a câmera
            else:
                self._start_camera()
        connection = asyncio.Queue(1)   # Contém somente uma posiçào na fila
        self._connections.add(connection)
        return connection

    def _terminate_stream_client(self, client):
        self._connections.discard(client)
        if len(self._connections)==0 and self._camera_running:
            # Mantém a câmera rodando por 2 segundos para evitar interromper
            #  por um reload de página
            self._stop_camera_task = self._loop.call_later(2, self._stop_camera)

    async def mjpeg_request(self, request):
        my_boundary = 'image-boundary'
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace;boundary={}'.format(my_boundary)
            }
        )
        await response.prepare(request)
        connection = self._create_new_stream_client()
        try:
            self._connections.add(connection)
            while True:
                try:
                    frame = await connection.get()
                    connection.task_done()
                    if frame is None:
                        # Encerramento de serviço
                        #   FIXME: queue.shutdown() é mais limpo, mas só existe em python 3.13
                        break
                    with MultipartWriter('image/jpeg', boundary=my_boundary) as mpwriter:
                        mpwriter.append(frame, {'Content-Type': 'image/jpeg'})
                        await mpwriter.write(response, close_boundary=False)
                except ConnectionResetError:
                    # Desconexão
                    break
                await response.write(b"\r\n")
        finally:
            # Remove conexão
            self._terminate_stream_client(connection)

    # Responde a uma solicitação de gravação
    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                #grava o último quadro
                frame = self._last_frame
                if frame is None:
                    return
                with tempfile.NamedTemporaryFile(suffix=".jpg", prefix ="captura", delete=False, dir=self._capture_path) as file:
                    print("Capturando em " + file.name)
                    file.write(frame)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                print('conexão ws encerrada com erro %s' %
                    ws.exception())
        return ws

    async def root_handler(self, request):
        return aiohttp.web.Response(text=request.app['root_pag'], content_type="text/html")

    async def inicializa_tarefas(self, app):
        self._loop = asyncio.get_running_loop()
        self._output = CallbackWriter(lambda buff : self._loop.call_soon_threadsafe(self.process_new_frame, buff))

    async def encerra_tarefas(self, app):
        print("Shutdown")
        # Encerra conexões
        for queue in self._connections:
            # Esvazia todas as filas, adiciona
            #   None para marcar término de servico
            # FIXME: Vide comentário sobre queue.shutdown() acima
            while not queue.empty():
                queue.get_nowait()
                queue.task_done()
            queue.put_nowait(None)
        self._connections = set()
        self._stop_camera()

def main():
    porta_servico = "8080"
    endereco_servidor = None
    diretorio_capturas = "./capturas"
    try:
      opts, args = getopt.getopt(sys.argv[1:],"hp:e:d:")
    except getopt.GetoptError:
      print('camera_stream.py -d <diretório de captura> -p <porta do servidor mjpeg> -e <endereco externo do servidor>')
      sys.exit(2)
    for opt, arg in opts:
        print(opt)
        if opt == '-h':
            print('camera_stream.py -d <diretório de captura> -p <porta do servidor mjpeg> -e <endereco externo do servidor>')
            sys.exit()
        elif opt in ("-p",):
            porta_servico = arg
        elif opt in ("-e",):
            endereco_servidor = arg
        elif opt in ("-d",):
            diretorio_capturas = arg

    if endereco_servidor == None:
        endereco_servidor = "0.0.0.0"

    print("Endereço do servidor mjpeg: " + endereco_servidor)
    print("Porta do servidor mjpeg: " + porta_servico)
    print("Diretório para imagens capturadas: " + diretorio_capturas)

    # Verifica se o diretório existe, caso não exista, cria
    if not os.path.isdir(diretorio_capturas):
        print("Diretório para imagens capturadas não existe, criando.")
        try:
            os.mkdir(diretorio_capturas)
        except:
            print("Impossível criar o caminho " + diretorio_capturas + ". Encerrando.")
            return -1


    srv = StreamingAndCaptureServer(diretorio_capturas)
    srv.run(endereco_servidor, porta_servico)

    return 0

if __name__ == '__main__':
    sys.exit(main())
