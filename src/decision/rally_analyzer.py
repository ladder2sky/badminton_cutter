from typing import List, Dict, Tuple

class ClipSegment:
    def __init__(self, start: float, end: float, score: float, type: str = "rally"):
        self.start_time = start
        self.end_time = end
        self.score = score
        self.type = type # 'rally', 'serve', 'cheer'

class RallyAnalyzer:
    """
    负责综合视频和音频信息，判定回合 (Rally) 的开始与结束。
    """
    def __init__(self, config):
        self.config = config
        self.min_rally_duration = 3.0 # 最小回合时长
        self.max_serve_interval = 5.0 # 发球准备最大间隔
        self.debug_file = None
        self.hit_map = {}

    def initialize_debug_writer(self, csv_path: str, hit_events: List[float]):
        self.debug_csv_path = csv_path
        self.hit_map = {int(t*30): True for t in hit_events}
        
        try:
            self.debug_file = open(self.debug_csv_path, "w", buffering=1) # Line buffered
            self.debug_file.write("frame_id,time,ball_x,ball_y,ball_conf,player_count,is_hit\n")
            print(f"[RallyAnalyzer] Debug CSV initialized at {self.debug_csv_path}")
        except PermissionError:
            import time
            self.debug_csv_path = f"debug_frames_{int(time.time())}.csv"
            print(f"[RallyAnalyzer] Warning: Permission Denied. Saving to {self.debug_csv_path} instead.")
            try:
                self.debug_file = open(self.debug_csv_path, "w", buffering=1)
                self.debug_file.write("frame_id,time,ball_x,ball_y,ball_conf,player_count,is_hit\n")
            except Exception as e:
                print(f"[RallyAnalyzer] Error initializing debug CSV: {e}")
        except Exception as e:
             print(f"[RallyAnalyzer] Error initializing debug CSV: {e}")

    def write_debug_frame(self, event: Dict):
        if self.debug_file:
            try:
                fid = event['frame_id']
                t = event['time']
                pos = event['ball_pos']
                pc = event['player_count']
                conf = event.get('ball_conf', 0.0)
                
                bx, by = pos if pos else (-1, -1)
                is_hit = 1 if int(t*30) in self.hit_map else 0
                
                self.debug_file.write(f"{fid},{t:.3f},{bx},{by},{conf:.4f},{pc},{is_hit}\n")
            except Exception as e:
                print(f"[RallyAnalyzer] Error writing debug frame: {e}")

    def close_debug_writer(self):
        if self.debug_file:
            self.debug_file.close()
            self.debug_file = None
            print(f"[RallyAnalyzer] Debug CSV closed.")
        
    def analyze(self, 
                hit_events: List[float], 
                video_events: List[Dict], # 包含每帧是否有球、球员位置
                cheer_events: List[Tuple[float, float]]) -> List[ClipSegment]:
        """
        核心逻辑重构 (Visual-Based State Machine)：
        1. 以羽毛球的飞行轨迹 (Ball Sequence) 为核心线索。
        2. 提取连续的羽毛球检测片段。
        3. 合并中断时间较短的片段 (Merge Gaps)。
        4. 验证片段有效性 (时长、位移、球员在场)。
        5. 利用击球声 (Audio) 辅助延长回合结束时间 (处理杀球导致球丢失的情况)。
        """
        if not video_events:
            return []
            
        print("[RallyAnalyzer] Starting vision-based analysis...")
        
        # Note: Debug CSV is now written incrementally in main.py via write_debug_frame
        if self.debug_file:
            self.close_debug_writer()
        
        # 1. 提取羽毛球轨迹片段
        sequences = self._extract_ball_sequences(video_events)
        print(f"[RallyAnalyzer] Found {len(sequences)} raw ball sequences.")
        
        # 2. 合并片段 (填补视觉丢失的空隙)
        merged_sequences = self._merge_sequences(sequences, max_gap=1.0)
        print(f"[RallyAnalyzer] Merged into {len(merged_sequences)} candidate sequences.")
        
        rallies = []
        
        for seq in merged_sequences:
            start_time, end_time, ball_positions = seq
            
            # 3. 验证片段
            
            # 3.1 时长检查
            duration = end_time - start_time
            if duration < 1.5: # 至少持续1.5秒 (发球+飞行)
                print(f"[Debug] Rejected sequence {start_time:.2f}-{end_time:.2f}: Too short ({duration:.2f}s)")
                continue
                
            # 3.2 运动幅度检查 (Bounding Box)
            # 过滤掉固定不动的噪点 (比如墙上的白点)
            xs = [p[0] for p in ball_positions]
            ys = [p[1] for p in ball_positions]
            if not xs: continue
            
            bbox_w = max(xs) - min(xs)
            bbox_h = max(ys) - min(ys)
            
            # 阈值：宽或高至少有一个超过 30 像素 (假设画面 640x360)
            if bbox_w < 30 and bbox_h < 30:
                print(f"[Debug] Rejected sequence {start_time:.2f}-{end_time:.2f}: Static noise (w={bbox_w}, h={bbox_h})")
                continue
                
            # 3.3 球员在场检查
            # 统计该时间段内的平均球员数量
            avg_players = self._get_avg_players(video_events, start_time, end_time)
            if avg_players < 0.5: # 允许偶尔遮挡，但平均要有人
                print(f"[Debug] Rejected sequence {start_time:.2f}-{end_time:.2f}: No players (avg={avg_players:.2f})")
                continue
                
            # 4. 优化边界 & 音频辅助
            
            # 4.1 修正开始时间
            # 发球前通常有准备动作，向前回溯 0.5 秒
            final_start = max(0, start_time - 0.6)
            
            # 4.2 修正结束时间 (Audio Extension)
            # 如果在视觉结束后的短时间内有击球声 (如重杀)，则延长回合
            final_end = end_time
            last_hit = self._find_last_hit_in_range(hit_events, end_time, end_time + 1.5)
            if last_hit:
                final_end = last_hit + 1.0 # 击球后延时
            else:
                final_end = end_time + 1.0 # 默认落地后延时
                
            # 计算分数 (简单逻辑：时长 + 击球数)
            hits_in_rally = [h for h in hit_events if final_start <= h <= final_end]
            score = (final_end - final_start) * 1.0 + len(hits_in_rally) * 2.0
            
            rallies.append(ClipSegment(final_start, final_end, score))
            
        # 5. 合并重叠的 Rally (由于前后扩展时间，原本独立的片段可能会重叠)
        rallies = self._merge_overlapping_rallies(rallies)
        
        print(f"[RallyAnalyzer] Final valid rallies: {len(rallies)}")
        return rallies

    def _merge_overlapping_rallies(self, rallies):
        """
        合并时间重叠的 Rally 片段。
        """
        if not rallies:
            return []
            
        # 按开始时间排序
        sorted_rallies = sorted(rallies, key=lambda r: r.start_time)
        merged = []
        
        current_rally = sorted_rallies[0]
        
        for next_rally in sorted_rallies[1:]:
            # 如果当前 Rally 的结束时间 大于 下一个 Rally 的开始时间 (或者非常接近)
            # 这里的阈值可以设为 0，表示只要重叠就合并
            if current_rally.end_time >= next_rally.start_time:
                # 合并
                new_end = max(current_rally.end_time, next_rally.end_time)
                new_score = max(current_rally.score, next_rally.score) # 取最高分
                
                # 更新 current_rally
                current_rally = ClipSegment(current_rally.start_time, new_end, new_score)
                print(f"[RallyAnalyzer] Merged overlapping rallies: {current_rally.start_time:.2f}-{current_rally.end_time:.2f}")
            else:
                # 没有重叠，保存当前，切换到下一个
                merged.append(current_rally)
                current_rally = next_rally
                
        merged.append(current_rally)
        return merged


    def _extract_ball_sequences(self, video_events):
        """
        从逐帧事件中提取连续的球检测片段。
        Returns: list of (start_time, end_time, [(x,y), ...])
        """
        sequences = []
        current_seq_positions = []
        current_seq_start = None
        last_frame_time = None
        
        for event in video_events:
            t = event['time']
            pos = event['ball_pos']
            
            if pos is not None:
                if current_seq_start is None:
                    current_seq_start = t
                current_seq_positions.append(pos)
                last_frame_time = t
            else:
                # 球丢失
                # 如果当前有序列，且丢失时间超过阈值 (这里先只提取纯连续片段，合并留给下一步)
                if current_seq_start is not None:
                    # 一个纯连续片段结束 (只要断一帧就算断，合并交给 _merge_sequences)
                    sequences.append((current_seq_start, last_frame_time, current_seq_positions))
                    current_seq_positions = []
                    current_seq_start = None
                    
        # 处理最后一个
        if current_seq_start is not None:
            sequences.append((current_seq_start, last_frame_time, current_seq_positions))
            
        return sequences

    def _merge_sequences(self, sequences, max_gap=1.0):
        """
        合并时间间隔较短的片段。
        """
        if not sequences:
            return []
            
        merged = []
        if not sequences:
            return []
            
        # curr: [start, end, positions]
        curr_start, curr_end, curr_pos = sequences[0]
        
        for i in range(1, len(sequences)):
            next_start, next_end, next_pos = sequences[i]
            
            gap = next_start - curr_end
            if gap <= max_gap:
                # 合并
                curr_end = next_end
                curr_pos.extend(next_pos) # 注意：中间gap的pos是空的，这里直接连起来，计算bbox没问题
            else:
                merged.append((curr_start, curr_end, curr_pos))
                curr_start, curr_end, curr_pos = next_start, next_end, next_pos
                
        merged.append((curr_start, curr_end, curr_pos))
        return merged

    def _get_avg_players(self, video_events, start_time, end_time):
        """
        计算指定时间段内的平均球员数量。
        """
        count = 0
        total_players = 0
        
        # 优化：可以使用二分查找找到 start_index，这里简单遍历 (假设 events 有序)
        # 考虑到 video_events 可能很大，最好二分。但 Python list slice 也很快。
        # 简单遍历优化：
        
        for event in video_events:
            t = event['time']
            if t < start_time:
                continue
            if t > end_time:
                break
            
            total_players += event['player_count']
            count += 1
            
        return total_players / count if count > 0 else 0

    def _find_last_hit_in_range(self, hit_events, start_time, end_time):
        """
        查找指定时间范围内最后一次击球声的时间。
        """
        last_hit = None
        for h in hit_events:
            if start_time <= h <= end_time:
                last_hit = h
            elif h > end_time:
                break
        return last_hit
