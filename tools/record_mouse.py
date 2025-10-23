import argparse
import csv
import os
import threading
import time
from typing import Optional, Tuple

from pynput import mouse, keyboard
import pyautogui

HEADER = ["time_ms", "x", "y", "target_x", "target_y", "target_w", "start_flag", "end_flag"]

class Recorder:
    def __init__(self, out_path: str, fps: int = 240, target_w: float = 40.0):
        self.out_path = out_path
        self.fps = max(30, fps)
        self.target_w = target_w
        self.recording = False
        self.quit_flag = False
        self.segment_buffer = []
        self.segment_start_ts = 0.0
        self.target_xy: Optional[Tuple[float, float]] = None
        self.lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self):
        os.makedirs(os.path.dirname(self.out_path) or ".", exist_ok=True)
        if not os.path.exists(self.out_path) or os.path.getsize(self.out_path) == 0:
            with open(self.out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(HEADER)

    def start_segment(self):
        with self.lock:
            if self.recording:
                return
            self.recording = True
            self.segment_buffer = []
            self.target_xy = None
            self.segment_start_ts = time.time()
            print("[REC] Segment started. Move the mouse and left-click to finish...")

    def end_segment(self):
        print("[REC] Ending segment...", self.lock)
        
        with self.lock:
            print("[REC] Ending segment233333...", self.lock)
            if not self.recording:
                return
            if not self.segment_buffer:
                print("[REC] Segment empty. Ignored.")
                self.recording = False
                return
            if self.target_xy is None:
                last = self.segment_buffer[-1]
                self.target_xy = (last[1], last[2])
                print("[REC] No click detected, using last position as target.")

            tx, ty = self.target_xy
            n = len(self.segment_buffer)
            with open(self.out_path, "a", newline="") as f:
                w = csv.writer(f)
                for i, row in enumerate(self.segment_buffer):
                    time_ms, x, y = row
                    sf = 1 if i == 0 else 0
                    ef = 1 if i == n - 1 else 0
                    w.writerow([time_ms, x, y, tx, ty, self.target_w, sf, ef])
            print(f"[REC] Segment saved: {n} samples -> {self.out_path}")
            self.recording = False
            self.segment_buffer = []    
            self.target_xy = None

    def on_click(self, x, y, button, pressed):
        if not pressed:
            return
        if button == mouse.Button.left:
            with self.lock:
                if self.recording and self.target_xy is None:
                    self.target_xy = (float(x), float(y))
                    print(f"[REC] Target captured at click: {x}, {y})")
                    
    def polling_loop(self):
        dt = 1.0 / self.fps
        while not self.quit_flag:
            t0 = time.time()
            if self.recording:
                x, y = pyautogui.position()
                now_ms = int(round((time.time() - self.segment_start_ts) * 1000.0))
               
                print(f"[REC] Target captured at mouse move: {x}, {y})")
                self.segment_buffer.append([now_ms, float(x), float(y)])
                if self.target_xy is not None and len(self.segment_buffer) >= 2:
                    pass
            elapsed = time.time() - t0
            to_sleep = dt - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
           
            if self.recording and self.target_xy is not None and len(self.segment_buffer) >= 2:
                self.end_segment()

    def run(self):
        mouse_listener = mouse.Listener(on_click=self.on_click)
        mouse_listener.start()

        poller = threading.Thread(target=self.polling_loop, daemon=True)
        poller.start()



        print("Hotkeys: Ctrl+Alt+S to start, then left-click to finish. Ctrl+Alt+Q to quit.")
        print(f"Output: {self.out_path}, FPS={self.fps}, target_w={self.target_w}")

        with keyboard.GlobalHotKeys({
            '<ctrl>+s': self.start_segment,
            '<ctrl>+<alt>+q': self._quit_action
        }) as h:
            h.join()

        mouse_listener.stop()
        mouse_listener.join(timeout=1.0)

    def _quit_action(self):
        print("[REC] Quit signal.")
        self.quit_flag = True
        if self.recording:
            self.end_segment()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="data/train.csv")
    parser.add_argument("--fps", type=int, default=240)
    parser.add_argument("--target-w", type=float, default=40.0)
    args = parser.parse_args()

    rec = Recorder(out_path=args.out, fps=args.fps, target_w=args.target_w)
    rec.run()

if __name__ == "__main__":
    try:
        pyautogui.FAILSAFE = False
    except Exception:
        pass
    main()
