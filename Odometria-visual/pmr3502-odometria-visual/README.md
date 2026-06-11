# pmr3502-odometria-visual (Tarefa 7.7)

Serviço de estimativa de posição do robô por odometria visual. Clone de
`gitlab.uspdigital.usp.br/thiago/pmr3502-odometria-visual` com a classe
`ProcessamentoVisao` (em `odometria_visual.py`) completada.

## Execução (na Raspberry Pi do robô)

```bash
python3 servidor.py            # interface web em http://<robo>:8084/
```

Abra a URL no navegador para ver a estimativa de posição acumulada. Movimente o
robô (p.ex. com a telepresença da seção 2.5) sobre uma superfície **com textura**
— pisos homogêneos não funcionam.

## Pré-requisitos

- OpenCV compilado **com SURF** (`cv2.xfeatures2d.SURF_create`), como no cap. 6.
- `camera_calibration_results.npz` do cap. 6. É procurado, nesta ordem, em:
  `$CALIB_PATH`, ao lado deste arquivo, e em `../pmr3502-camera-calibration/`.
  Se necessário: `CALIB_PATH=/caminho/results.npz python3 servidor.py`.
- `picamera2`, `aiohttp`.

## Como funciona `ProcessamentoVisao`

- `__init__`: carrega `mtx`, `dist`, `s` (retificação) da calibração; monta a
  matriz de reprojeção `Q` (1 px/mm, área visível x'∈[0,600] mm,
  y'∈[-480,480] mm) e o mapa via `initUndistortRectifyMap`; constrói uma máscara
  da área visível (região cujo mapa cai dentro do quadro original, erodida); cria
  o detector SURF com o `treshold` recebido (ajuste para ~100 pontos/quadro).
- `primeiroQuadro`: reprojeta, detecta SURF e guarda kp/descritores.
- `estimaMovimento`: reprojeta, detecta SURF, casa com o quadro anterior
  (`knnMatch` k=2 + teste de Lowe 0.7). Exige ≥4 associações (alerta se <8) e
  ≥5 inliers do RANSAC. Estima `L*` com `estimateAffinePartial2D`
  (anterior→atual = mundo em relação ao robô), aplica a eq. 7.6
  `L' = Q⁻¹ L* Q` e **inverte** `L'` para obter o movimento do robô em relação
  ao mundo. Retorna `(δx', δy', δθ')`; em qualquer falha retorna `(0,0,0)`, mas
  sempre guarda os descritores do quadro atual.

A acumulação de pose (eq. da Tarefa item 1) já está em `EstimadorPosicao`.

> Nota: a matemática de `estimaMovimento` (Q, eq. 7.6 e inversão) foi validada
> numa simulação geométrica (recupera δx', δy', δθ' exatos). O pipeline completo
> só pôde ser testado parcialmente fora do robô, pois exige câmera + SURF.
> Ajuste `treshold` (3000 por padrão, em `EstimadorPosicao.__init__`) para obter
> ~100 pontos por quadro.
