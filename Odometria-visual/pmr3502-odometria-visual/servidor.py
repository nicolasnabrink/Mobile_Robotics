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
from aiohttp import web
import sys
import subprocess
import argparse
import numpy as np
from odometria_visual import cria_estimador_posicao
import time
import concurrent
import time

class RobotPositionService():

    def __init__(self, estimador, app, endereco_servidor, porta_servidor):
        self._estimador_posicao = estimador
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
        self._loop = None

    def run(self):
        web.run_app(self._app, host=self._endereco_servidor, port=self._porta_servidor, shutdown_timeout=0.2)

    async def _pagina(self, request):
        return web.FileResponse('./static/showposition.html')

    async def _inicializa_tarefas(self, app):
        self._loop = asyncio.get_running_loop()
        self._worker_task = self._loop.run_in_executor(None, self._worker)

    async def _encerra_tarefas(self, app):
        self._keep_alive = False
        await self._worker_task
        self._worker_task = None

    # Responde a uma conexão web socket
    async def _websocket_handler(self, request):
        print("new connection")
        messages = Queue(5)
        self._connections.add(messages)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            while self._keep_alive and not ws.closed:
                try:
                    from_client = await ws.receive()
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
            raise e
        self._connections.remove(messages)

    # Invoca o estimador de visão
    #   Deve rodar em uma thread separada
    def _worker(self):
        visao = self._estimador_posicao
        while self._keep_alive:
            dados = list(visao.atualizaPosicao())
            self._loop.call_soon_threadsafe(self._nova_mensagem, dados)
            # Evita starving se o processamento de imagem
            #   consumir mais tempo do que a taxa de quadros
            #   da câmera
            time.sleep(0)

    def _nova_mensagem(self, dados):
        for connection in self._connections:
            if not connection.full():   # Backlogged
                connection.put_nowait(dados)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', help="Endereço externo do servidor")
    parser.add_argument('-p', help="Porta do servidor", default="8084")

    args = parser.parse_args()
    endereco_servidor = args.e
    porta_servico = args.p

    if endereco_servidor == None:
        endereco_servidor = "0.0.0.0"

    print("Endereço do servidor: " + endereco_servidor)
    print("Porta do servidor: " + porta_servico)


    print("Url do servico: http://" + endereco_servidor + ":" + porta_servico + "/")

    with cria_estimador_posicao() as estimador:
        RobotPositionService(estimador, web.Application(), endereco_servidor, int(porta_servico)).run()

    return 0


if __name__ == '__main__':
    sys.exit(main())
