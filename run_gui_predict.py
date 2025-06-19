import sys
import torch
import numpy as np
# PyQt5 대신 PySide6 사용
from PySide6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog, QLabel
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt  # matplotlib.pyplot 명시적으로 임포트

from policy import NestingDNN
from environment import NestingEnv

# --- 실제 학습 파라미터 (반드시 학습과 동일하게 맞출 것) ---
SHEET_SIZE = (400, 400)
AREA_REWARD_SCALE = 0.00001
DISTANCE_REWARD_SCALE = 0.5
DISTANCE_THRESHOLD = 5.0
DISTANCE_EPSILON = 1e-6
BOUNDARY_REWARD_SCALE = 2
MAX_STEPS = 50
MAX_FD_LEN = 20
MAX_POLY_LEN = 100
EXISTING_SHAPE_COLS = [
    'area', 'perimeter', 'num_vertices', 'aspect_ratio', 'compactness', 'circularity',
    'convexity', 'hole_count', 'bounding_box_diagonal', 'shape_density', 'width', 'height',
    'convex_hull_area', 'centroid_x', 'centroid_y', 'is_convex'
]

# --- DXF 전처리 함수 (학습 파라미터 반영) ---
def preprocess_dxf_to_piece_data(dxf_path, scaler):
    try:
        # 기존의 문자열 변환 과정을 건너뛰고 ezdxf로 직접 파일 읽기
        from utils.polygon_utils import ezdxf_to_polygon
        import ezdxf

        # 직접 파일을 읽어서 ezdxf 문서 객체 생성
        try:
            dxf_doc = ezdxf.readfile(dxf_path)
            print(f"DXF 파일 '{dxf_path}' 직접 로드 성공")
        except Exception as e:
            print(f"ezdxf로 파일 직접 읽기 실패: {str(e)}, 대체 방법 시도")
            # 파일 직접 읽기 실패 시 기존 방식 시도
            from utils.dxf_utils import dxfstr_to_polygon
            with open(dxf_path, 'r', encoding='utf-8') as f:
                dxf_content = f.read()
            dxf_doc = dxfstr_to_polygon(dxf_content)

        polygon = ezdxf_to_polygon(dxf_doc)

        from shapely.geometry import Polygon
        from math import sqrt
        def extract_polygon_features(polygon):
            if polygon is None:
                return {col: 0 for col in EXISTING_SHAPE_COLS}
            area = polygon.area
            perimeter = polygon.length
            num_vertices = len(polygon.exterior.coords) - 1 if len(polygon.exterior.coords) > 0 else 0
            minx, miny, maxx, maxy = polygon.bounds
            width = maxx - minx
            height = maxy - miny
            aspect_ratio = width / height if height > 0 else 0
            convex_hull_area = polygon.convex_hull.area
            convexity = area / convex_hull_area if convex_hull_area != 0 else 0
            centroid = polygon.centroid
            compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
            circularity = (area * 4 * np.pi) / (perimeter ** 2) if perimeter > 0 else 0
            return {
                'area': area, 'width': width, 'height': height, 'perimeter': perimeter,
                'num_vertices': num_vertices, 'aspect_ratio': aspect_ratio, 'compactness': compactness,
                'circularity': circularity, 'convexity': convexity, 'convex_hull_area': convex_hull_area,
                'centroid_x': centroid.x, 'centroid_y': centroid.y, 'is_convex': int(convexity >= 0.98),
                'hole_count': len(polygon.interiors),
                'bounding_box_diagonal': sqrt(width**2 + height**2),
                'shape_density': area / (width * height) if width * height else 0
            }

        shape_features = extract_polygon_features(polygon)
        shape_features_vec = np.array([shape_features[col] for col in EXISTING_SHAPE_COLS], dtype=np.float32)

        from scipy.fft import fft
        def extract_fourier_descriptors(polygon, num_coeffs=MAX_FD_LEN):
            if polygon is None: return [0] * num_coeffs
            coords = polygon.exterior.coords
            if len(coords) < 3: return [0] * num_coeffs
            complex_boundary = np.array([complex(x, y) for x, y in coords])
            dft_coeffs = fft(complex_boundary)
            effective_num_coeffs = min(num_coeffs, (len(dft_coeffs) - 1) // 2)
            fourier_descriptors = np.abs(dft_coeffs[1:effective_num_coeffs + 1])
            padded = list(fourier_descriptors) + [0] * (num_coeffs - effective_num_coeffs)
            if effective_num_coeffs > 0 and np.abs(dft_coeffs[1]) > 1e-9:
                normalized = np.array(padded) / np.abs(dft_coeffs[1])
            else:
                normalized = np.array(padded)
            return normalized.tolist()

        fourier_vec = extract_fourier_descriptors(polygon, num_coeffs=MAX_FD_LEN)

        coords = polygon.exterior.coords if polygon else []
        poly_vec = [coord for point in coords for coord in point]
        poly_vec = poly_vec + [0] * (MAX_POLY_LEN - len(poly_vec))

        combined_features = np.hstack((shape_features_vec, fourier_vec))

        # 스케일러 관련 오류 처리 개선
        try:
            if hasattr(scaler, 'transform'):
                features_scaled = scaler.transform([combined_features])[0]
            else:
                print(f"스케일러가 transform 메소드를 가지고 있지 않습니다. 기본값 사용.")
                features_scaled = combined_features  # 스케일링 건너뜀
        except Exception as e:
            print(f"특성 스케일링 중 오류 발생: {str(e)}. 원본 특성 사용.")
            features_scaled = combined_features  # 오류 시 원본 특성 사용

        polygon_coords_np = np.array(poly_vec, dtype=np.float32).reshape(-1, 2)
        non_zero_rows_mask = (np.abs(polygon_coords_np) > 1e-9).any(axis=1)
        last_non_zero_index = np.where(non_zero_rows_mask)[0]
        if last_non_zero_index.size > 0:
            last_non_zero_index = last_non_zero_index[-1]
            mask_np = np.zeros(polygon_coords_np.shape[0], dtype=bool)
            mask_np[:last_non_zero_index + 1] = True
        else:
            mask_np = np.zeros(polygon_coords_np.shape[0], dtype=bool)

        piece_data = [{
            'polygon_coords': polygon_coords_np.astype(np.float32),
            'polygon_mask': mask_np,
            'features': features_scaled.astype(np.float32)
        }]
        return piece_data
    except Exception as e:
        print(f"{dxf_path} 처리 중 오류: {str(e)}")
        return []  # 빈 리스트 반환하여 이 파일 건너뛰기

class NestingPredictor(QWidget):
    def __init__(self, policy_net, scaler):
        super().__init__()
        self.policy_net = policy_net
        self.scaler = scaler
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('DXF Nesting Predictor')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()

        self.label = QLabel('DXF 파일을 최대 20개까지 선택하세요.')
        layout.addWidget(self.label)

        self.btn = QPushButton('DXF 파일 선택 및 예측')
        self.btn.clicked.connect(self.open_files)
        layout.addWidget(self.btn)

        self.canvas = FigureCanvas(Figure(figsize=(6, 6)))
        layout.addWidget(self.canvas)

        self.setLayout(layout)

    def open_files(self):
        fnames, _ = QFileDialog.getOpenFileNames(self, 'DXF 파일 선택', '', 'DXF Files (*.dxf)')
        if fnames:
            if len(fnames) > 20:
                fnames = fnames[:20]
                self.label.setText(f"20개까지만 처리합니다. (선택된 파일 수: {len(fnames)})")
            else:
                self.label.setText(f"선택된 파일: {len(fnames)}개")

            # 여러 DXF 파일을 전처리하여 piece_data 리스트로 만듦
            piece_data = []
            for fname in fnames:
                try:
                    piece_data.extend(preprocess_dxf_to_piece_data(fname, self.scaler))
                except Exception as e:
                    print(f"{fname} 처리 중 오류: {e}")

            if not piece_data:
                self.label.setText("유효한 도형이 없습니다.")
                return

            env = NestingEnv(
                sheet_size=SHEET_SIZE,
                pieces_data=piece_data,
                heuristic_type=None,
                heuristic_step_size=5.0,
                area_reward_scale=AREA_REWARD_SCALE,
                distance_reward_scale=DISTANCE_REWARD_SCALE,
                distance_threshold=DISTANCE_THRESHOLD,
                distance_epsilon=DISTANCE_EPSILON,
                boundary_reward_scale=BOUNDARY_REWARD_SCALE,
                max_steps=MAX_STEPS
            )
            observation_np, info = env.reset()
            done = False
            total_reward = 0
            step = 0
            coords_list = []
            while not done:
                observation = torch.tensor(observation_np, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = self.policy_net.get_action_and_value(observation)
                action_np = action.squeeze(0).cpu().numpy()
                next_observation_np, reward, terminated, truncated, info = env.step(action_np)
                total_reward += reward
                step += 1
                observation_np = next_observation_np
                done = terminated or truncated
                coords_list = [c for c in env.placed_pieces_coords]
            self.plot_result(coords_list, env.sheet_size)
            self.label.setText(f"예측 완료: 총 리워드={total_reward}, 스텝={step}, 배치 성공 조각 수={info.get('placed_pieces_count', 0)}")

    def plot_result(self, coords_list, sheet_size):
        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.set_xlim(0, sheet_size[0])
        ax.set_ylim(0, sheet_size[1])
        ax.set_aspect('equal')

        # 시트 영역 경계선 표시
        sheet_rect = plt.Rectangle((0, 0), sheet_size[0], sheet_size[1],
                                  fill=False, edgecolor='black', linestyle='-', linewidth=2)
        ax.add_patch(sheet_rect)

        # 각 도형을 채워진 폴리곤으로 표시
        for i, coords in enumerate(coords_list):
            actual_coords = coords[(np.abs(coords).sum(axis=1) > 1e-6)]
            if actual_coords.shape[0] > 2:
                # 도형을 채우기 위해 첫 점을 마지막에 추가하여 닫힌 경로 생성
                poly = np.vstack([actual_coords, actual_coords[0]])

                # 다양한 색상으로 도형 표시 (인덱스에 따라 색상 변경)
                color_idx = i % 10  # 10개의 색상 순환
                colors = ['lightblue', 'lightgreen', 'salmon', 'lightpink', 'lightyellow',
                          'lightcoral', 'lightseagreen', 'lightskyblue', 'lightsteelblue', 'plum']

                # 채워진 폴리곤 추가
                polygon_patch = plt.Polygon(poly,
                                           fill=True,
                                           facecolor=colors[color_idx],
                                           edgecolor='blue',
                                           alpha=0.7,
                                           linewidth=1.5)
                ax.add_patch(polygon_patch)

                # 도형 중심에 번호 표시 (선택사항)
                centroid = actual_coords.mean(axis=0)
                ax.text(centroid[0], centroid[1], str(i+1),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=10,
                        color='black',
                        fontweight='bold')

        ax.set_title("배치 결과")
        ax.grid(True, linestyle='--', alpha=0.6)  # 그리드 추가
        self.canvas.draw()

def main():
    # 기존 checkpoint_path 수정
    checkpoint_path = ("./models/checkpoint_episode.pth")  # 최신 모델 중 하나를 선택

    import joblib
    from sklearn.preprocessing import StandardScaler

    # 스케일러 파일이 존재하는지 확인하고, 없으면 새로 생성
    scaler_path = "./models/scaler.pkl"
    try:
        scaler = joblib.load(scaler_path)
        print("기존 스케일러를 불러왔습니다.")
    except FileNotFoundError:
        print("스케일러 파일이 없어 기본 스케일러를 사용합니다.")
        scaler = StandardScaler()
        # 스케일러 저장 디렉토리 생성 (이미 있으면 넘어감)
        import os
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        # 기본 스케일러 저장
        joblib.dump(scaler, scaler_path)

    # PyTorch 2.6 이상 버전에서는 weights_only=False 옵션을 명시적으로 설정
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # 체크포인트에서 feature_extractor.0.weight의 크기를 확인하여 입력 차원 결정
    model_state_dict = checkpoint['model_state_dict']
    feature_extractor_weight = model_state_dict['feature_extractor.0.weight']
    input_dim = feature_extractor_weight.shape[1]  # 체크포인트 모델의 입력 차원 사용
    print(f"체크포인트 모델의 입력 차원: {input_dim}")

    policy_net = NestingDNN(input_dim)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    policy_net.eval()

    app = QApplication(sys.argv)
    win = NestingPredictor(policy_net, scaler)
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()