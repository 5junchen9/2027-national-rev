# -*- coding: utf-8 -*-
"""可配置串口监视器：实时显示收发数据，并支持文本/HEX 发送。"""

import argparse
import queue
import threading
from datetime import datetime


LINE_ENDINGS = {"无": b"", "LF": b"\n", "CR": b"\r", "CRLF": b"\r\n"}


def build_payload(text, mode, encoding, line_ending):
    if mode == "HEX":
        cleaned = text.replace(",", " ").replace("0x", "").replace("0X", "")
        payload = bytes.fromhex(cleaned)
    else:
        payload = text.encode(encoding)
    return payload + LINE_ENDINGS[line_ending]


def format_payload(data, mode, encoding):
    hex_text = data.hex(" ").upper()
    text = data.decode(encoding, errors="replace").replace("\r", "\\r").replace("\n", "\\n")
    if mode == "HEX":
        return hex_text
    if mode == "文本+HEX":
        return "{}    [{}]".format(text, hex_text)
    return text


def self_check():
    assert build_payload("你好", "文本", "utf-8", "CRLF") == "你好".encode("utf-8") + b"\r\n"
    assert build_payload("01 0xA2,ff", "HEX", "utf-8", "无") == b"\x01\xA2\xFF"
    assert format_payload(b"A\r\n", "文本+HEX", "utf-8") == r"A\r\n    [41 0D 0A]"
    print("串口监视器自检通过")


def run_gui():
    import tkinter as tk
    from tkinter import messagebox, ttk

    try:
        import serial
        from serial.tools import list_ports
    except ImportError:
        messagebox.showerror("缺少依赖", "请先安装 pyserial：\npython3 -m pip install pyserial")
        return False

    class SerialMonitor:
        def __init__(self, root):
            self.root = root
            self.serial = None
            self.reader = None
            self.stop_event = threading.Event()
            self.events = queue.Queue()

            root.title("串口收发监视器")
            root.geometry("900x620")
            root.minsize(720, 480)

            settings = ttk.LabelFrame(root, text="串口设置", padding=8)
            settings.pack(fill="x", padx=10, pady=(10, 5))

            self.port = tk.StringVar()
            self.baud = tk.StringVar(value="115200")
            self.data_bits = tk.StringVar(value="8")
            self.parity = tk.StringVar(value="无")
            self.stop_bits = tk.StringVar(value="1")
            self.encoding = tk.StringVar(value="utf-8")
            self.display_mode = tk.StringVar(value="文本+HEX")

            fields = (
                ("端口", self.port, (), 13),
                ("波特率", self.baud, ("1200", "2400", "4800", "9600", "19200", "38400", "57600", "115200"), 10),
                ("数据位", self.data_bits, ("5", "6", "7", "8"), 5),
                ("校验", self.parity, ("无", "奇", "偶"), 5),
                ("停止位", self.stop_bits, ("1", "1.5", "2"), 6),
                ("编码", self.encoding, ("utf-8", "gbk", "ascii"), 8),
                ("显示", self.display_mode, ("文本+HEX", "文本", "HEX"), 10),
            )
            self.port_box = None
            for column, (label, variable, values, width) in enumerate(fields):
                ttk.Label(settings, text=label).grid(row=0, column=column, sticky="w")
                state = "normal" if label in ("端口", "波特率") else "readonly"
                box = ttk.Combobox(settings, textvariable=variable, values=values, width=width, state=state)
                box.grid(row=1, column=column, padx=(0, 6), sticky="ew")
                if label == "端口":
                    self.port_box = box

            self.refresh_button = ttk.Button(settings, text="刷新", command=self.refresh_ports)
            self.refresh_button.grid(row=1, column=len(fields), padx=(2, 6))
            self.open_button = ttk.Button(settings, text="打开串口", command=self.toggle_port)
            self.open_button.grid(row=1, column=len(fields) + 1)

            log_frame = ttk.LabelFrame(root, text="收发记录", padding=8)
            log_frame.pack(fill="both", expand=True, padx=10, pady=5)
            self.log = tk.Text(log_frame, wrap="word", state="disabled", font=("Consolas", 10))
            scrollbar = ttk.Scrollbar(log_frame, command=self.log.yview)
            self.log.configure(yscrollcommand=scrollbar.set)
            self.log.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            self.log.tag_configure("RX", foreground="#126E00")
            self.log.tag_configure("TX", foreground="#005BBB")
            self.log.tag_configure("ERR", foreground="#B00020")

            sender = ttk.LabelFrame(root, text="发送", padding=8)
            sender.pack(fill="x", padx=10, pady=(5, 10))
            self.send_mode = tk.StringVar(value="文本")
            self.line_ending = tk.StringVar(value="无")
            self.input = ttk.Entry(sender)
            self.input.pack(side="left", fill="x", expand=True)
            self.input.bind("<Return>", lambda _event: self.send())
            ttk.Combobox(sender, textvariable=self.send_mode, values=("文本", "HEX"), width=6,
                         state="readonly").pack(side="left", padx=(8, 4))
            ttk.Combobox(sender, textvariable=self.line_ending, values=tuple(LINE_ENDINGS), width=6,
                         state="readonly").pack(side="left", padx=4)
            self.send_button = ttk.Button(sender, text="发送", command=self.send, state="disabled")
            self.send_button.pack(side="left", padx=(4, 0))
            ttk.Button(sender, text="清空记录", command=self.clear_log).pack(side="left", padx=(8, 0))

            self.status = tk.StringVar(value="未连接")
            ttk.Label(root, textvariable=self.status, anchor="w").pack(fill="x", padx=12, pady=(0, 8))

            self.refresh_ports()
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
            self.root.after(50, self.process_events)

        def refresh_ports(self):
            ports = [item.device for item in list_ports.comports()]
            self.port_box["values"] = ports
            if ports and not self.port.get():
                self.port.set(ports[0])
            self.status.set("发现 {} 个串口".format(len(ports)))

        def toggle_port(self):
            if self.serial:
                self.close_port()
            else:
                self.open_port()

        def open_port(self):
            if not self.port.get().strip():
                messagebox.showwarning("缺少端口", "请选择或输入串口，例如 COM3 或 /dev/ttyUSB0。")
                return
            parity = {"无": serial.PARITY_NONE, "奇": serial.PARITY_ODD, "偶": serial.PARITY_EVEN}
            try:
                self.serial = serial.Serial(
                    port=self.port.get().strip(),
                    baudrate=int(self.baud.get()),
                    bytesize=int(self.data_bits.get()),
                    parity=parity[self.parity.get()],
                    stopbits=float(self.stop_bits.get()),
                    timeout=0.1,
                    write_timeout=1,
                )
            except (ValueError, KeyError, serial.SerialException) as error:
                self.serial = None
                messagebox.showerror("打开失败", str(error))
                return

            self.stop_event.clear()
            self.reader = threading.Thread(target=self.read_loop, args=(self.serial,), daemon=True)
            self.reader.start()
            self.open_button.configure(text="关闭串口")
            self.send_button.configure(state="normal")
            self.status.set("已连接：{} @ {}".format(self.serial.port, self.serial.baudrate))
            self.append_log("SYS", "串口已打开")

        def close_port(self):
            current = self.serial
            self.serial = None
            self.stop_event.set()
            if current:
                current.close()
            self.open_button.configure(text="打开串口")
            self.send_button.configure(state="disabled")
            self.status.set("未连接")
            self.append_log("SYS", "串口已关闭")

        def read_loop(self, current):
            while not self.stop_event.is_set() and current.is_open:
                try:
                    data = current.read(current.in_waiting or 1)
                    if data:
                        self.events.put(("RX", data))
                except (OSError, serial.SerialException) as error:
                    if not self.stop_event.is_set():
                        self.events.put(("ERR", "读取失败：{}".format(error)))
                    break

        def send(self):
            if not self.serial:
                return
            try:
                payload = build_payload(
                    self.input.get(), self.send_mode.get(), self.encoding.get(), self.line_ending.get()
                )
                if not payload:
                    return
                sent = self.serial.write(payload)
                self.serial.flush()
                if sent != len(payload):
                    raise serial.SerialTimeoutException("只发送了 {}/{} 字节".format(sent, len(payload)))
                self.events.put(("TX", payload))
            except (LookupError, UnicodeError, ValueError, OSError, serial.SerialException) as error:
                messagebox.showerror("发送失败", str(error))

        def process_events(self):
            while True:
                try:
                    kind, value = self.events.get_nowait()
                except queue.Empty:
                    break
                if kind in ("RX", "TX"):
                    value = format_payload(value, self.display_mode.get(), self.encoding.get())
                self.append_log(kind, value)
                if kind == "ERR" and self.serial:
                    self.close_port()
            self.root.after(50, self.process_events)

        def append_log(self, kind, value):
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.log.configure(state="normal")
            self.log.insert("end", "[{}] {:>3}  {}\n".format(stamp, kind, value), kind)
            self.log.configure(state="disabled")
            self.log.see("end")

        def clear_log(self):
            self.log.configure(state="normal")
            self.log.delete("1.0", "end")
            self.log.configure(state="disabled")

        def on_close(self):
            if self.serial:
                self.close_port()
            self.root.destroy()

    root = tk.Tk()
    SerialMonitor(root)
    root.mainloop()
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="只运行自检，不打开窗口和串口")
    args = parser.parse_args()
    if args.check:
        self_check()
        return True
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
