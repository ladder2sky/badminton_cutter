import collections

class StaticFilter:
    """
    过滤器：用于识别并屏蔽静态的误检目标（如顶棚灯光、地胶反光）。
    原理：真正的羽毛球是高速运动的，如果一个目标在连续多帧内位置几乎不变，则视为干扰。
    """
    def __init__(self, history_size=30, dist_threshold=5.0):
        self.history = collections.deque(maxlen=history_size)
        self.dist_threshold = dist_threshold

    def is_static(self, pos):
        if not pos: 
            self.history.clear() # Reset on loss
            return False
        
        self.history.append(pos)
        if len(self.history) < 10: 
            return False
        
        # 计算历史轨迹的极差 (Range)
        xs = [p[0] for p in self.history]
        ys = [p[1] for p in self.history]
        
        range_x = max(xs) - min(xs)
        range_y = max(ys) - min(ys)
        
        # 如果 X 和 Y 方向的波动都极小，说明是静止物体
        if range_x < self.dist_threshold and range_y < self.dist_threshold:
            return True
        
        return False
