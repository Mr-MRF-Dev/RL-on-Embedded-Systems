"""Live PC-side visualizer for the Cart-Pole and Grid-World DQN demo sketches
(cartpole_dqn_demo.ino / grid_dqn_demo.ino).

Reads the demo stream from the board over serial, detects which demo is running
from the line tag ("CP" or "GRID"), and animates the agent next to a stats
panel (episode, step, action, rewards, outcome).

Usage:
    pip install pyserial
    python rl_serial_visualizer.py

Select the serial port and click Connect. The Pause / Resume / Next Episode
buttons send 'p' / 'g' / 'n' to the board.
"""

import math
import queue
import threading
import tkinter as tk
from tkinter import ttk

import serial
import serial.tools.list_ports

BAUD_RATE = 115200

# Defaults used until a CP_INFO / GRID_INFO line arrives from the board
DEFAULT_X_THRESHOLD = 2.4
DEFAULT_THETA_THRESHOLD = 0.2094   # 12 degrees in rad
DEFAULT_CP_MAX_STEPS = 200
DEFAULT_GRID_N = 30
DEFAULT_GRID_MAX_STEPS = 180

CARTPOLE_ACTIONS = {0: "Push Left", 1: "Push Right"}
GRID_ACTIONS = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}

STAT_LABELS = [
    ("episode", "Episode"), ("step", "Step"), ("action", "Action"),
    ("reward", "Reward (step)"), ("total_reward", "Cumulative reward"),
    ("avg_return", "Avg return (last 10 ep)"), ("status", "Status"),
]


class SerialReader(threading.Thread):
    """Reads lines from the serial port on a background thread."""

    def __init__(self, port, baud, line_queue, status_queue):
        super().__init__(daemon=True)
        self.port = port
        self.baud = baud
        self.line_queue = line_queue
        self.status_queue = status_queue
        self._stop_flag = threading.Event()
        self._ser = None

    def run(self):
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.2)
        except Exception as exc:
            self.status_queue.put(("error", f"Could not open {self.port}: {exc}"))
            return

        self.status_queue.put(("connected", self.port))
        while not self._stop_flag.is_set():
            try:
                raw = self._ser.readline()
            except Exception as exc:
                self.status_queue.put(("error", f"Serial read error: {exc}"))
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if line:
                self.line_queue.put(line)

        try:
            self._ser.close()
        except Exception:
            pass
        self.status_queue.put(("disconnected", self.port))

    def send_command(self, char):
        if self._ser is not None and self._ser.is_open:
            try:
                self._ser.write(char.encode("ascii"))
            except Exception:
                pass

    def stop(self):
        self._stop_flag.set()


def build_stats_row(parent):
    """Row of (label, value) boxes; returns {key: StringVar}."""
    row = ttk.Frame(parent)
    row.pack(fill="x", padx=10)
    variables = {}
    for col, (key, label_text) in enumerate(STAT_LABELS):
        var = tk.StringVar(value="-")
        variables[key] = var
        box = ttk.Frame(row)
        box.grid(row=0, column=col, padx=8, pady=(4, 8), sticky="w")
        ttk.Label(box, text=label_text, font=("Segoe UI", 9)).pack(anchor="w")
        ttk.Label(box, textvariable=var, font=("Segoe UI", 13, "bold")).pack(anchor="w")
    return variables


class CartPoleView(ttk.Frame):
    """Cart-and-pole animation plus stats."""

    def __init__(self, parent):
        super().__init__(parent)
        self.x_threshold = DEFAULT_X_THRESHOLD
        self.theta_threshold = DEFAULT_THETA_THRESHOLD
        self.max_steps = DEFAULT_CP_MAX_STEPS
        self.recent_returns = []

        self.canvas = tk.Canvas(self, width=760, height=340, background="#101418",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.vars = build_stats_row(self)
        self._draw(0.0, 0.0)

    def set_info(self, x_threshold, theta_threshold, max_steps):
        self.x_threshold = x_threshold
        self.theta_threshold = theta_threshold
        self.max_steps = max_steps

    def update_state(self, episode, step, x, x_dot, theta, theta_dot,
                     action, reward, total_reward, done, result):
        self.vars["episode"].set(episode)
        self.vars["step"].set(step)
        self.vars["action"].set(CARTPOLE_ACTIONS.get(action, str(action)))
        self.vars["reward"].set(f"{reward:+.2f}")
        self.vars["total_reward"].set(f"{total_reward:+.2f}")

        if done:
            self.vars["status"].set(
                "SOLVED (balanced full episode)" if result == "SOLVED"
                else "FAILED (fell / out of bounds)"
            )
            self.recent_returns = (self.recent_returns + [total_reward])[-10:]
            self.vars["avg_return"].set(f"{sum(self.recent_returns) / len(self.recent_returns):.1f}")
        else:
            self.vars["status"].set("running")

        self._draw(x, theta)

    def _draw(self, x, theta):
        c = self.canvas
        c.delete("all")
        w, h = 760, 340
        track_y = h - 70
        margin = 40

        c.create_text(w / 2, 22, text="Cart-Pole -- on-device DQN inference",
                      fill="#c8ccd0", font=("Segoe UI", 11, "bold"))

        c.create_line(margin, track_y, w - margin, track_y, fill="#3a4048", width=3)
        for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
            px = margin + (w - 2 * margin) * (frac + 1) / 2
            c.create_line(px, track_y - 6, px, track_y + 6, fill="#3a4048")

        frac = max(-1.2, min(1.2, x / self.x_threshold))
        cart_x = margin + (w - 2 * margin) * (frac + 1) / 2
        cart_w, cart_h = 70, 34

        pole_len = 110
        tip_x = cart_x + pole_len * math.sin(theta)
        tip_y = (track_y - cart_h / 2) - pole_len * math.cos(theta)
        c.create_line(cart_x, track_y - cart_h / 2, tip_x, tip_y,
                      fill="#e0a030", width=6, capstyle="round")
        c.create_oval(tip_x - 6, tip_y - 6, tip_x + 6, tip_y + 6,
                      fill="#e0a030", outline="")

        c.create_rectangle(cart_x - cart_w / 2, track_y - cart_h,
                           cart_x + cart_w / 2, track_y,
                           fill="#3d7ea6", outline="#7fc7ff", width=2)
        for wheel_dx in (-cart_w / 3, cart_w / 3):
            c.create_oval(cart_x + wheel_dx - 8, track_y - 4,
                          cart_x + wheel_dx + 8, track_y + 12,
                          fill="#20242a", outline="#555")


class GridView(ttk.Frame):
    """Grid-world animation plus stats."""

    def __init__(self, parent):
        super().__init__(parent)
        self.grid_n = DEFAULT_GRID_N
        self.max_steps = DEFAULT_GRID_MAX_STEPS
        self.recent_returns = []
        self.trail = []

        self.canvas = tk.Canvas(self, width=520, height=520, background="#101418",
                                highlightthickness=0)
        self.canvas.pack(padx=10, pady=10)
        self.vars = build_stats_row(self)
        self._draw(0, 0)

    def set_info(self, grid_n, max_steps):
        self.grid_n = grid_n
        self.max_steps = max_steps
        self.trail = []

    def update_state(self, episode, step, row, col, action, reward,
                     total_reward, done, result):
        self.vars["episode"].set(episode)
        self.vars["step"].set(step)
        self.vars["action"].set(GRID_ACTIONS.get(action, str(action)))
        self.vars["reward"].set(f"{reward:+.2f}")
        self.vars["total_reward"].set(f"{total_reward:+.2f}")

        if step <= 1:
            self.trail = []
        self.trail.append((row, col))

        if done:
            self.vars["status"].set(
                "GOAL reached" if result == "GOAL" else "TIMEOUT (goal not reached)"
            )
            self.recent_returns = (self.recent_returns + [total_reward])[-10:]
            self.vars["avg_return"].set(f"{sum(self.recent_returns) / len(self.recent_returns):.1f}")
        else:
            self.vars["status"].set("running")

        self._draw(row, col)

    def _draw(self, row, col):
        c = self.canvas
        c.delete("all")
        w, h = 520, 520
        margin = 10
        n = max(1, self.grid_n)
        cell = (min(w, h) - 2 * margin) / n

        for i in range(n + 1):
            px = margin + i * cell
            c.create_line(px, margin, px, margin + n * cell, fill="#2a2f36")
            c.create_line(margin, px, margin + n * cell, px, fill="#2a2f36")

        gx0 = margin + (n - 1) * cell
        gy0 = margin + (n - 1) * cell
        c.create_rectangle(gx0, gy0, gx0 + cell, gy0 + cell, fill="#2f6b3a", outline="")
        c.create_text(gx0 + cell / 2, gy0 + cell / 2, text="G", fill="#c8ffd0",
                      font=("Segoe UI", max(8, int(cell / 2)), "bold"))

        if len(self.trail) > 1:
            points = []
            for (r, cix) in self.trail:
                points.extend([margin + (cix + 0.5) * cell, margin + (r + 0.5) * cell])
            c.create_line(*points, fill="#3d7ea6", width=max(2, int(cell / 8)))

        ax = margin + (col + 0.5) * cell
        ay = margin + (row + 0.5) * cell
        radius = max(4, cell * 0.32)
        c.create_oval(ax - radius, ay - radius, ax + radius, ay + radius,
                      fill="#e0a030", outline="")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RL On-Device Inference -- Live Visualizer")
        self.geometry("820x680")

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.line_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.reader = None

        self._build_top_bar()

        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.views = {
            "cartpole": CartPoleView(self.container),
            "grid": GridView(self.container),
        }
        self.active_view = None
        self._placeholder = None
        self._show_placeholder()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_queues)

    # ---------------- top bar ----------------
    def _build_top_bar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=10, pady=8)

        ttk.Label(bar, text="Port:").pack(side="left")
        self.port_var = tk.StringVar()
        self.port_combo = ttk.Combobox(bar, textvariable=self.port_var, width=18,
                                       state="readonly")
        self.port_combo.pack(side="left", padx=(4, 4))
        self._refresh_ports()

        ttk.Button(bar, text="Refresh", command=self._refresh_ports).pack(side="left")
        self.connect_btn = ttk.Button(bar, text="Connect", command=self._toggle_connect)
        self.connect_btn.pack(side="left", padx=(8, 16))

        ttk.Button(bar, text="Pause", command=lambda: self._send("p")).pack(side="left")
        ttk.Button(bar, text="Resume", command=lambda: self._send("g")).pack(
            side="left", padx=(4, 4))
        ttk.Button(bar, text="Next Episode", command=lambda: self._send("n")).pack(
            side="left")

        self.conn_status_var = tk.StringVar(value="Not connected")
        ttk.Label(bar, textvariable=self.conn_status_var).pack(side="right")

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports and not self.port_var.get():
            self.port_var.set(ports[0])

    def _toggle_connect(self):
        if self.reader is None:
            port = self.port_var.get()
            if not port:
                self.conn_status_var.set("Select a port first")
                return
            self.reader = SerialReader(port, BAUD_RATE, self.line_queue, self.status_queue)
            self.reader.start()
            self.connect_btn.config(text="Disconnect")
        else:
            self.reader.stop()
            self.reader = None
            self.connect_btn.config(text="Connect")
            self.conn_status_var.set("Not connected")

    def _send(self, char):
        if self.reader is not None:
            self.reader.send_command(char)

    def _on_close(self):
        if self.reader is not None:
            self.reader.stop()
        self.destroy()

    # ---------------- view switching ----------------
    def _show_placeholder(self):
        self._placeholder = ttk.Label(
            self.container,
            text="Connect to the board and flash cartpole_dqn_demo.ino\n"
                 "or grid_dqn_demo.ino -- the matching view appears automatically\n"
                 "once data starts arriving.",
            justify="center", font=("Segoe UI", 12),
        )
        self._placeholder.pack(expand=True)

    def _show_view(self, name):
        if self._placeholder is not None:
            self._placeholder.pack_forget()
            self._placeholder = None
        for view_name, view in self.views.items():
            if view_name != name:
                view.pack_forget()
        self.views[name].pack(fill="both", expand=True)
        self.active_view = name

    # ---------------- queue polling / line parsing ----------------
    def _poll_queues(self):
        while not self.status_queue.empty():
            kind, payload = self.status_queue.get_nowait()
            if kind == "connected":
                self.conn_status_var.set(f"Connected: {payload}")
            elif kind == "disconnected":
                self.conn_status_var.set("Not connected")
            elif kind == "error":
                self.conn_status_var.set(payload)
                self.reader = None
                self.connect_btn.config(text="Connect")

        while not self.line_queue.empty():
            self._handle_line(self.line_queue.get_nowait())

        self.after(50, self._poll_queues)

    def _handle_line(self, line):
        parts = line.split(",")
        tag = parts[0]

        try:
            if tag == "CP_INFO" and len(parts) >= 4:
                self.views["cartpole"].set_info(float(parts[1]), float(parts[2]), int(parts[3]))

            elif tag == "GRID_INFO" and len(parts) >= 3:
                self.views["grid"].set_info(int(parts[1]), int(parts[2]))

            elif tag == "CP" and len(parts) >= 11:
                if self.active_view != "cartpole":
                    self._show_view("cartpole")
                self.views["cartpole"].update_state(
                    episode=int(parts[1]), step=int(parts[2]),
                    x=float(parts[3]), x_dot=float(parts[4]),
                    theta=float(parts[5]), theta_dot=float(parts[6]),
                    action=int(parts[7]), reward=float(parts[8]),
                    total_reward=float(parts[9]), done=bool(int(parts[10])),
                    result=parts[11] if len(parts) > 11 else "",
                )

            elif tag == "GRID" and len(parts) >= 9:
                if self.active_view != "grid":
                    self._show_view("grid")
                self.views["grid"].update_state(
                    episode=int(parts[1]), step=int(parts[2]),
                    row=int(parts[3]), col=int(parts[4]),
                    action=int(parts[5]), reward=float(parts[6]),
                    total_reward=float(parts[7]), done=bool(int(parts[8])),
                    result=parts[9] if len(parts) > 9 else "",
                )
            # other lines (boot banners, debug text) are ignored
        except (ValueError, IndexError):
            # malformed or torn line (can happen right after boot)
            pass


def main():
    App().mainloop()


if __name__ == "__main__":
    main()
