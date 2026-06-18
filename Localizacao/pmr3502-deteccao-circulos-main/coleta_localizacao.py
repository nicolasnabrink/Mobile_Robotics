import asyncio

import json
import websockets


endereco_robo = "localhost"  # Substitua pelo endereço do seu robô
porta_servico_circulos = "8086"
porta_servico_imu = 1234


async def mostra_circulos():
    uri = f"ws://{endereco_robo}:{porta_servico_circulos}/wsctrl"
    async with websockets.connect(uri) as websocket:
        while True:
            await websocket.send("ack")
            dados = json.loads(await websocket.recv())
            cc = ""
            for x, y in dados["coordenadas"]:
                cc = f"{cc},{x},{y}"
            print(f"{dados['timestamp']},c,{cc}")


async def mostra_imu():
    reader, writer = await asyncio.open_connection(
        endereco_robo, porta_servico_imu)
    while True:
        data = await reader.read(32)
        ax = int.from_bytes(data[0:2], byteorder="big", signed=True)
        ay = int.from_bytes(data[2:4], byteorder="big", signed=True)
        az = int.from_bytes(data[4:6], byteorder="big", signed=True)
        wx = int.from_bytes(data[8:10], byteorder="big", signed=True)
        wy = int.from_bytes(data[10:12], byteorder="big", signed=True)
        wz = int.from_bytes(data[12:14], byteorder="big", signed=True)
        t = int.from_bytes(data[-8:], byteorder="little", signed=False)
        print(f"{t},i,{ax},{ay},{az},{wx},{wy},{wz}")


async def main():
    await asyncio.gather(mostra_circulos(), mostra_imu())
    return


if __name__ == "__main__":
    asyncio.run(main())
