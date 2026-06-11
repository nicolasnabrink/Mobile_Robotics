import cv2
import numpy as np
import traceback
from picamera2 import Picamera2
import os
import contextlib


def _carrega_calibracao():
    """Carrega os resultados de calibracao salvos no capitulo 6.

    Procura o arquivo camera_calibration_results.npz em CALIB_PATH (variavel de
    ambiente) ou em alguns caminhos relativos usuais. Retorna (mtx, dist, s),
    onde 's' eh a homografia de retificacao do solo (parametro R de
    initUndistortRectifyMap).
    """
    candidatos = []
    if os.environ.get('CALIB_PATH'):
        candidatos.append(os.environ['CALIB_PATH'])
    aqui = os.path.dirname(os.path.abspath(__file__))
    candidatos += [
        os.path.join(aqui, 'camera_calibration_results.npz'),
        os.path.join(aqui, '..', 'pmr3502-camera-calibration',
                     'camera_calibration_results.npz'),
        os.path.join(aqui, '..', '..', 'pmr3502-camera-calibration',
                     'camera_calibration_results.npz'),
    ]
    for caminho in candidatos:
        if caminho and os.path.exists(caminho):
            calib = np.load(caminho)
            print('Calibracao carregada de:', os.path.abspath(caminho))
            return calib['mtx'], calib['dist'], calib['s']
    raise FileNotFoundError(
        'camera_calibration_results.npz nao encontrado. Defina a variavel de '
        'ambiente CALIB_PATH apontando para o arquivo salvo no capitulo 6, ou '
        'coloque-o ao lado de odometria_visual.py. Tentado: ' +
        ', '.join(c for c in candidatos if c))


class ProcessamentoVisao():
    """Estima a movimentacao relativa do robo entre quadros sucessivos.

    A cada quadro a imagem capturada eh reprojetada para o solo (correcao de
    perspectiva, secao 6.9.1), pontos caracteristicos SURF sao detectados e
    associados aos do quadro anterior. A transformacao de similaridade entre os
    dois conjuntos de pontos (equacao 7.5) eh estimada por
    estimateAffinePartial2D e convertida para o referencial global pela equacao
    7.6 (L' = Q^-1 L* Q). Como essa transformacao descreve o movimento do
    *mundo* em relacao ao robo, ela eh invertida para obter o movimento do robo
    em relacao ao mundo, que eh o que estimaMovimento retorna.
    """

    # Parametros de reprojeccao do solo. Escala de 1 pixel/mm (suficiente,
    # segundo a apostila) maximizando a area visivel: x' de 0 a xmax mm a frente
    # do robo e y' de -0.8*xmax a +0.8*xmax (abertura lateral).
    PIXELS_POR_MM = 1.0
    XMAX_MM = 600.0

    # Numero minimo de associacoes para tentar estimar (4) e numero ideal (8).
    MIN_ASSOCIACOES = 4
    IDEAL_ASSOCIACOES = 8
    # Minimo de inliers que o RANSAC do estimateAffinePartial2D deve manter.
    MIN_INLIERS = 5
    # Razao do teste de Lowe.
    LOWE_RATIO = 0.7

    def __init__(self, largura, altura, treshold):
        self._largura = int(largura)
        self._altura = int(altura)

        mtx, dist, s_retificacao = _carrega_calibracao()

        # Matriz de reprojeccao Q (secao 7.6):
        #   Q = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
        # com s = PIXELS_POR_MM, c = (cx, cy) o ponto do referencial do robo que
        # cai no canto superior esquerdo da imagem reprojetada. Aqui cx = 0 e
        # cy = -ymax, de modo que y' vai de -ymax (topo) a +ymax (base).
        s = self.PIXELS_POR_MM
        xmax = self.XMAX_MM
        ymax = 0.8 * xmax
        cx = 0.0
        cy = -ymax
        self._Q = np.float32([
            [s, 0.0, -s * cx],
            [0.0, s, -s * cy],
            [0.0, 0.0, 1.0],
        ])
        self._Q_inv = np.linalg.inv(self._Q)

        largura_reproj = int(round(s * xmax))           # x' de 0 a xmax
        altura_reproj = int(round(s * 2.0 * ymax))       # y' de -ymax a +ymax
        dim = (largura_reproj, altura_reproj)

        # Mapa de reprojeccao do solo (mesma chamada da secao 6.9.1).
        self._map_x, self._map_y = cv2.initUndistortRectifyMap(
            mtx, dist, s_retificacao, self._Q, dim, cv2.CV_32FC1)

        # Mascara da area visivel: pixels da imagem reprojetada cujo mapa aponta
        # para dentro dos limites do quadro original. Isso elimina as bordas
        # pretas (BORDER_CONSTANT) e a maior parte dos artefatos de distorcao,
        # sem precisar da foto da folha branca da secao 7.6 (que tambem poderia
        # ser usada para gerar uma mascara mais fiel ao suporte da camera).
        valida = (
            (self._map_x >= 0) & (self._map_x <= self._largura - 1) &
            (self._map_y >= 0) & (self._map_y <= self._altura - 1)
        )
        mascara = np.where(valida, 255, 0).astype(np.uint8)
        # Erode para afastar os pontos das bordas (margem de seguranca).
        mascara = cv2.erode(mascara, np.ones((9, 9), np.uint8), iterations=2)
        self._mascara = mascara

        # Detector SURF. O treshold (limiar do determinante da Hessiana) deve
        # ser ajustado para ~100 pontos por quadro (sugestao da apostila).
        self._surf = cv2.xfeatures2d.SURF_create(treshold)

        # Casador forca-bruta com norma L2 (recomendada para SURF).
        self._matcher = cv2.BFMatcher(cv2.NORM_L2)

        # Pontos/descritores do quadro anterior.
        self._kp_anterior = None
        self._desc_anterior = None

    def _reprojeta_e_detecta(self, quadro):
        """Converte para cinza, reprojeta o solo e detecta pontos SURF.

        Retorna (keypoints, descritores). descritores pode ser None se nenhum
        ponto for encontrado.
        """
        if quadro is None:
            return [], None

        # A imagem da picamera2 pode vir com 3 ou 4 canais; para o SURF basta
        # tons de cinza (a ordem dos canais de cor eh irrelevante).
        if quadro.ndim == 3 and quadro.shape[2] == 4:
            quadro = quadro[:, :, :3]
        if quadro.ndim == 3:
            cinza = cv2.cvtColor(quadro, cv2.COLOR_BGR2GRAY)
        else:
            cinza = quadro

        reproj = cv2.remap(cinza, self._map_x, self._map_y,
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        kp, desc = self._surf.detectAndCompute(reproj, self._mascara)
        return kp, desc

    def primeiroQuadro(self, quadro):
        """Invocado no primeiro quadro: apenas guarda os pontos de referencia."""
        try:
            self._kp_anterior, self._desc_anterior = \
                self._reprojeta_e_detecta(quadro)
        except Exception:
            traceback.print_exc()
            self._kp_anterior, self._desc_anterior = None, None

    def estimaMovimento(self, quadro):
        """Estima (dx', dy', dtheta') do robo em relacao ao quadro anterior.

        Retorna sempre uma tupla de 3 floats. Em caso de falha (poucos pontos,
        poucas associacoes ou poucos inliers) retorna (0, 0, 0), mas em todos os
        casos armazena os descritores do quadro atual para a proxima iteracao.
        """
        movimento = (0.0, 0.0, 0.0)
        kp_atual, desc_atual = [], None
        try:
            kp_atual, desc_atual = self._reprojeta_e_detecta(quadro)
            movimento = self._estima(kp_atual, desc_atual)
        except Exception:
            traceback.print_exc()
            movimento = (0.0, 0.0, 0.0)
        finally:
            # Guarda os pontos do quadro atual para a proxima iteracao,
            # aconteca o que acontecer.
            self._kp_anterior = kp_atual
            self._desc_anterior = desc_atual
        return movimento

    def _estima(self, kp_atual, desc_atual):
        # Precisamos de descritores nos dois quadros (e ao menos 2 no anterior
        # para o knnMatch com k=2).
        if (desc_atual is None or self._desc_anterior is None or
                len(desc_atual) < 2 or len(self._desc_anterior) < 2):
            print('AVISO: descritores insuficientes para casar quadros.')
            return (0.0, 0.0, 0.0)

        # query = quadro atual, train = quadro anterior.
        pares = self._matcher.knnMatch(desc_atual, self._desc_anterior, k=2)

        # Teste de Lowe para eliminar associacoes ambiguas.
        boas = []
        for par in pares:
            if len(par) == 2:
                m, n = par
                if m.distance < self.LOWE_RATIO * n.distance:
                    boas.append(m)

        if len(boas) < self.MIN_ASSOCIACOES:
            print('AVISO: apenas %d associacoes boas (< %d). Sem estimativa.'
                  % (len(boas), self.MIN_ASSOCIACOES))
            return (0.0, 0.0, 0.0)
        if len(boas) < self.IDEAL_ASSOCIACOES:
            print('AVISO: apenas %d associacoes boas (ideal >= %d).'
                  % (len(boas), self.IDEAL_ASSOCIACOES))

        # m.queryIdx -> kp_atual; m.trainIdx -> kp_anterior.
        pts_atual = np.float32([kp_atual[m.queryIdx].pt for m in boas])
        pts_anterior = np.float32([self._kp_anterior[m.trainIdx].pt
                                   for m in boas])

        # estimateAffinePartial2D(from, to): to ~ L* @ from. Mapeando os pontos
        # do quadro anterior para o atual obtemos como os pontos do solo se
        # movem na imagem, ou seja, o movimento do *mundo* em relacao ao robo,
        # no espaco de coordenadas da imagem (matriz L* da equacao 7.5).
        L_estrela, mascara = cv2.estimateAffinePartial2D(pts_anterior, pts_atual)
        if L_estrela is None:
            print('AVISO: estimateAffinePartial2D nao convergiu.')
            return (0.0, 0.0, 0.0)

        n_inliers = int(mascara.sum()) if mascara is not None else 0
        if n_inliers < self.MIN_INLIERS:
            print('AVISO: RANSAC manteve apenas %d associacoes (< %d).'
                  % (n_inliers, self.MIN_INLIERS))
            return (0.0, 0.0, 0.0)

        # L* (2x3) -> 3x3 homogenea.
        L_estrela_h = np.vstack([L_estrela, [0.0, 0.0, 1.0]])

        # Equacao 7.6: transformacao no espaco global = Q^-1 L* Q.
        # (mundo em relacao ao robo, agora em coordenadas globais/mm).
        L_linha = self._Q_inv @ L_estrela_h @ self._Q

        # O que queremos eh o movimento do robo em relacao ao mundo: o inverso.
        T_robo = np.linalg.inv(L_linha)

        dx = float(T_robo[0, 2])
        dy = float(T_robo[1, 2])
        dtheta = float(np.arctan2(T_robo[1, 0], T_robo[0, 0]))
        return (dx, dy, dtheta)


# Cria um estimador de posicao.
#   Este gerador eh decorado com contextmanager para garantir
#   que o estado da camera seja resetado na saida de escopo
@contextlib.contextmanager
def cria_estimador_posicao():
    # Atribui nivel "erro" para libcamera e picamera
    Picamera2.set_logging(Picamera2.ERROR)
    os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
    try:
        with Picamera2(tuning=os.environ.get('LIBCAMERA_RPI_TUNING_FILE', None)) as camera:
            camera.configure(camera.create_still_configuration(main={"size": (1296, 972)}))
            camera.start()
            yield EstimadorPosicao(camera)
            camera.stop()
    finally:
        pass


class EstimadorPosicao():

      def __init__(self, picam):
            self._x = 0.0
            self._y = 0.0
            self._t = 0.0
            self._picam = picam
            # Pega primeiro quadro
            quadro = self._picam.capture_array("main")
            self._processamento = ProcessamentoVisao(quadro.shape[1], quadro.shape[0], 3000)
            self._processamento.primeiroQuadro(quadro)

      def atualizaPosicao(self):
            img = self._picam.capture_array("main")

            if img is None:
                return (0.0, 0.0, 0.0)

            # Estima o movimento
            mov_x, mov_y, mov_t = self._processamento.estimaMovimento(img)

            # Atualiza a posicao
            self._x += mov_x*np.cos(self._t) - mov_y*np.sin(self._t)
            self._y += mov_y*np.cos(self._t) + mov_x*np.sin(self._t)
            self._t += mov_t

            return (self._x, self._y, self._t)
