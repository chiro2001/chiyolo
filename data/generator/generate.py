import os
import asyncio
import time
import math
import cv2
import random
import numpy as np
from tqdm import trange


# 取角点的外框
def vec2rect(vec) -> tuple:
    return (min([pos[0] for pos in vec]), min([pos[1] for pos in vec])), (max([pos[0] for pos in vec]), max([pos[1] for pos in vec]))


# 角点转极坐标
def vec2polar(vec) -> list:
    rect = vec2rect(vec)
    center = np.array(((rect[0][0] + rect[1][0]) // 2,
                       (rect[0][1] + rect[1][1]) // 2))
    # print("rect", rect, 'center', center)
    polar = []
    for v in vec:
        d = v - center
        alpha = math.atan((d[1] / d[0]) if d[0] !=
                          0 else (d[1] / (d[0] + 1e-8)))
        r = math.sqrt(d[0] ** 2 + d[1] ** 2)
        polar.append((r, alpha))
    return polar


class Enegy:
    class State:
        """
        速度目标函数为：spd=0.785*sin（1.884*t)+1.305
        其中，spd的单位为rad/s，t的单位为s
        每次大能量机关进入可激活状态时，t 重置为零。
        """

        def __init__(self) -> None:
            self.angle = 0
            self.enabled = False
            self.fans = [0 for _ in range(5)]
            self.timestamp_start = time.time()
            self.timestamp_last: float = None

        def reset(self):
            self.enabled = False
            self.fans = [0 for _ in range(5)]
            self.timestamp_start = time.time()
            self.timestamp_last: float = None

        def get_timer(self) -> float:
            return time.time() - self.timestamp_start

        def get_speed(self) -> float:
            return 0.785 * math.sin(1.884 * self.get_timer()) + 1.305

        def update_angle(self) -> float:
            if self.timestamp_last is None:
                self.timestamp_last = self.get_timer()
                return self.angle
            timestamp = self.get_timer()
            self.angle += self.get_speed() * (timestamp - self.timestamp_last)
            if self.angle >= math.pi * 2:
                self.angle -= math.pi * 2
            self.timestamp_last = timestamp
            return self.angle

        def disable(self):
            self.reset()

        def enable(self):
            self.disable()
            self.enabled = True
            self.fans[0] = 1

        def hit(self) -> bool:
            if not self.enabled:
                return False
            available = []
            for i in range(5):
                if self.fans[i] == 0:
                    available.append(i)
            cnt = -1
            for i in range(5):
                if self.fans[i] == 1:
                    cnt = i
                    break
            if cnt == -1:
                return False
            if len(available) == 0:
                self.fans[cnt] = 2
                return True
            # print('available', available)
            self.fans[cnt] = 2
            selected = random.randint(0, len(available) - 1)
            # print('select', selected)
            self.fans[available[selected]] = 1
            return False

    IMAGE_DIR: str = './data/generator/images'

    def __init__(self, draw_boarder: bool = False) -> None:
        self.resources = self.get_resources(draw_boarder=draw_boarder)
        self.state = Enegy.State()

    def get_polly(self, m: np.ndarray) -> list:
        return [np.array(np.array([*pos, 1]).T @ m, dtype=np.int64)
                for pos in self.resources['info']['fan']['poly']]
    
    def get_ms(self, angle: float):
        return self.get_rotate_mat(angle, self.resources['info']['center'])
    
    def get_ms_rect(self, ms: np.ndarray):
        return np.array([*ms, [0, 0, 1]])

    def get_resources(self, image_dir: str = IMAGE_DIR, draw_boarder: bool = True):
        images = os.listdir(image_dir)
        images = {filename.split('.')[0]: cv2.imread(os.path.join(
            image_dir, filename), -1) for filename in images if filename.endswith('.png')}
        # masks = {key: cv2.cvtColor(images[key][:, :, 3], cv2.COLOR_GRAY2RGB) for key in images.keys()}
        masks = {key: images[key][:, :, 3] for key in images.keys()}
        masked = {key: cv2.bitwise_and(
            images[key], images[key], mask=masks[key]) for key in masks.keys()}
        # cv2.imshow("masked", masked['base'])
        # cv2.waitKey(0)
        info = {
            'center': (514, 335),
            'fan': {
                'poly': [
                    (462, 20),
                    (462, 81), (504, 81),
                    (499, 283), (525, 283),
                    (528, 81), (564, 81),
                    (564, 20)
                ],
                'center': (512, 47)
            }
        }
        if draw_boarder:
            for i in range(3):
                pos_last = None
                for pos in info['fan']['poly']:
                    if pos_last is None:
                        pos_last = pos
                        continue
                    cv2.line(masked['fan%d' % i], pos, pos_last,
                             (0, 255, 0, 0), thickness=2)
                    pos_last = pos
                cv2.line(masked['fan%d' % i], info['fan']['poly'][-1],
                         info['fan']['poly'][0], (0, 255, 0, 255), thickness=3)
                cv2.circle(masked['fan%d' % i], info['fan']
                           ['center'], 10, (0, 255, 0, 255), -1)
        return {
            'images': masked,
            'info': info
        }

    @staticmethod
    def get_rotate_mat(angle: float, center: tuple, scale: float = 1.0) -> np.ndarray:
        alpha = scale * math.cos(angle)
        beta = scale * math.sin(angle)
        return np.array([
            [alpha, beta, (1 - alpha) * center[0] - beta * center[1]],
            [-beta, alpha, beta * center[0] + (1 - alpha) * center[1]]
        ])

    @staticmethod
    async def get_fan(fan_origin: np.ndarray, m: np.ndarray) -> np.ndarray:
        fan = fan_origin.copy()
        fan_dst = cv2.warpAffine(fan, m, dsize=(fan.shape[1], fan.shape[0]))
        return fan_dst
        # return 0
    
    def get_mats(self):
        angle = self.state.angle
        angle_delta = (2 * math.pi) / 5
        ms_all = [self.get_ms(angle + angle_delta * cnt) for cnt in range(5)]
        ms_rect_all = [self.get_ms_rect(ms_all[cnt]) for cnt in range(5)]
        poly_all = [self.get_polly(ms_rect_all[cnt].T) for cnt in range(5)]
        center_all = [np.array(np.array([*self.resources['info']['fan']['center'], 1]).T @ ms_rect_all[cnt].T, dtype=np.int64) for cnt in range(5)]
        return ms_all, ms_rect_all, poly_all, center_all

    def render(self, draw_boarder: bool = False) -> np.ndarray:
        base = self.resources['images']['base'].copy()
        if self.state.enabled:
            cv2.bitwise_and(self.resources['images']['center_r'], self.resources['images']
                            ['center_r'], mask=self.resources['images']['center_r'][:, :, 3], dst=base)
        else:
            cv2.bitwise_and(self.resources['images']['center'], self.resources['images']
                            ['center'], mask=self.resources['images']['center'][:, :, 3], dst=base)
        loop = asyncio.get_event_loop()
        # ms_all = [self.get_ms(angle + angle_delta * cnt) for cnt in range(5)]
        # ms_rect_all = [self.get_ms_rect(ms_all[cnt]) for cnt in range(5)]
        # poly_all = [self.get_polly(ms_rect_all[cnt].T) for cnt in range(5)]
        # center_all = [np.array(np.array([*self.resources['info']['fan']['center'], 1]
        #                                 ).T @ ms_rect_all[cnt].T, dtype=np.int64) for cnt in range(5)]
        ms_all, ms_rect_all, poly_all, center_all = self.get_mats()
        res = loop.run_until_complete(asyncio.wait([
            Enegy.get_fan(self.resources['images']
                          ['fan%d' % self.state.fans[cnt]], ms_all[cnt])
            for cnt in range(5)
        ]))
        results = [one.result() for one in list(res[0])]
        [cv2.bitwise_and(fan_dst, fan_dst, mask=fan_dst[:, :, 3], dst=base)
         for fan_dst in results]
        if draw_boarder:
            [[cv2.circle(base, tuple((pos[0], pos[1])), 1, (0, 255, 0), -1)
              for pos in poly_all[cnt]] for cnt in range(5)]
            [cv2.circle(base, tuple((center_all[cnt][0], center_all[cnt][1])),
                        3, (0, 255, 0), -1) for cnt in range(5)]
            [cv2.rectangle(base, *vec2rect(poly_all[cnt]), (0, 255, 0), 1)
             for cnt in range(5)]
        return base

    def show(self, wait_time: int = 0, exit_key=27, draw_boarder: bool = False) -> str:
        im = self.render(draw_boarder=draw_boarder)
        cv2.imshow("Enegy", im)
        key = cv2.waitKey(wait_time) & 0xFF
        if key == exit_key:
            print("Exit...")
            exit(0)
        return chr(key)


def generate(batch: int = 2):
    enegy = Enegy()
    if not os.path.exists("data"):
        os.mkdir("data")
    if not os.path.exists("data/simulator_dataset"):
        os.mkdir("data/simulator_dataset")
    if not os.path.exists("data/simulator_dataset/imgs"):
        os.mkdir("data/simulator_dataset/imgs")
    with open("data/simulator_dataset/simulator.txt", "w") as f:
        for cnt in trange(batch):
            ms_all, ms_rect_all, poly_all, center_all = enegy.get_mats()
            im = enegy.render(draw_boarder=False)
            # im_resized = cv2.resize(im, (800, 600))
            filename = f"img_r_{cnt}.png"
            # cv2.imshow("im_resized", im_resized)
            cv2.imshow("im_resized", im)

            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'e':
                if enegy.state.enabled:
                    enegy.state.disable()
                else:
                    enegy.state.enable()
            elif key == ' ':
                res = enegy.state.hit()
                if res:
                    print("Successfully enable Enegy")
            elif key == 'q':
                exit(0)
            
            # cv2.imwrite(os.path.join("data/simulator_dataset/imgs", filename), im_resized)
            cv2.imwrite(os.path.join("data/simulator_dataset/imgs", filename), im)
            line = f'{filename} '
            for i in range(5):
                rect = vec2rect(poly_all[i])
                header = ','.join(map(lambda x: str(x), rect[0])) + ',' + ','.join(map(lambda x: str(x), rect[1])) + ",0,"
                poly = poly_all[i]
                line += header + ','.join(map(lambda x: str(x), np.array([pos[:2] for pos in poly]).flatten().tolist())) + ' '
            line = line[:-1] + '\n'
            # print(line)
            f.write(line)
            enegy.state.update_angle()
    os.system("copy data\\simulator_dataset\\simulator.txt data\\simulator_dataset\\simulator-val.txt")
    os.system("copy data\\simulator_dataset\\simulator.txt data\\simulator_dataset\\simulator-train.txt")
    os.system("copy data\\simulator_dataset\\simulator.txt data\\simulator_dataset\\simulator-test.txt")

def main():
    enegy = Enegy()
    pause = False
    while True:
        if pause:
            key = chr(cv2.waitKey(1) & 0xFF)
        else:
            key = enegy.show(wait_time=1, exit_key=ord('q'), draw_boarder=True)
        if key == 'e':
            if enegy.state.enabled:
                enegy.state.disable()
            else:
                enegy.state.enable()
        elif key == ' ':
            res = enegy.state.hit()
            if res:
                print("Successfully enable Enegy")
        elif key == 'p':
            pause = not pause
        elif key == 'q':
            exit(0)
        if not pause:
            enegy.state.update_angle()
        # time.sleep(0.1)


if __name__ == '__main__':
    # main()

    # ve = [(0, 0), (100, 0), (100, 200), (0, 200)]
    # print(ve)
    # print(vec2polar(ve))

    generate()
