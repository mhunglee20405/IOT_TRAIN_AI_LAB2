# -*- coding: utf-8 -*-
"""
src/telegram_service.py  —  Dịch Vụ Telegram Bot Tích Hợp YOLO11n
================================================================
Quản lý tương tác Telegram Bot với pipeline nhận diện YOLO11n:
  - Tái sử dụng worker inference từ FastAPI/Web Dashboard (không tải thêm model YOLO).
  - Lệnh /start: Hướng dẫn sử dụng.
  - Lệnh /status: Thống kê số lượng mask, no_mask, unknown từ kết quả YOLO mới nhất.
  - Lệnh /photo: Gửi ảnh chụp mới nhất kèm bounding box và nhãn.
  - Lệnh /flash: Kích hoạt endpoint /flash trên ESP32-CAM.
  - Kiểm tra tính tươi mới của dữ liệu (stale detection threshold).
  - Phân quyền theo danh sách Telegram Chat ID được phép.
  - Phát hiện và cảnh báo xung đột HTTP 409 Conflict (khi firmware Arduino cũng đang poll cùng token).
"""

import io
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, List, Optional, Set, Tuple

import requests

# Đảm bảo in tiếng Việt / ký tự đặc biệt ổn định trên console Windows
if sys.platform == "win32" and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    except Exception:
        pass

# Thiết lập logging
logger = logging.getLogger("TelegramService")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class TelegramBotService:
    """
    Dịch vụ Telegram Bot polling thời gian thực, đọc dữ liệu trực tiếp từ ESP32PipelineWorker.
    """

    def __init__(
        self,
        token: str,
        allowed_chat_ids: Optional[List[str]] = None,
        worker: Any = None,
        stale_threshold: float = 5.0,
        api_base_url: str = "https://api.telegram.org",
    ):
        self.token = token.strip()
        if not self.token:
            raise ValueError("Telegram Bot Token không được để trống.")

        self.api_base_url = api_base_url.rstrip("/")
        self.api_url = f"{self.api_base_url}/bot{self.token}"
        self.worker = worker
        self.stale_threshold = float(stale_threshold)

        # Chuẩn hóa danh sách Chat ID được phép
        self.allowed_chat_ids: Set[str] = set()
        if allowed_chat_ids:
            for cid in allowed_chat_ids:
                clean_cid = str(cid).strip()
                if clean_cid:
                    self.allowed_chat_ids.add(clean_cid)

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.session: Optional[requests.Session] = None
        self.last_update_id = 0

    def is_chat_allowed(self, chat_id: Any) -> bool:
        """Kiểm tra Chat ID có quyền gửi lệnh tới bot hay không."""
        cid_str = str(chat_id).strip()
        if not self.allowed_chat_ids:
            logger.warning(f"Chưa cấu hình allowed_chat_ids. Tạm thời cho phép chat_id={cid_str}.")
            return True
        return cid_str in self.allowed_chat_ids

    def send_message(self, chat_id: Any, text: str, parse_mode: Optional[str] = "HTML") -> bool:
        """Gửi tin nhắn văn bản tới chat_id."""
        if not self.session:
            return False
        url = f"{self.api_url}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode

        try:
            res = self.session.post(url, json=payload, timeout=8.0)
            if res.status_code != 200:
                logger.error(f"Lỗi sendMessage (HTTP {res.status_code}): {res.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Ngoại lệ khi sendMessage tới {chat_id}: {e}")
            return False

    def send_photo(self, chat_id: Any, photo_bytes: bytes, caption: str = "") -> bool:
        """Gửi ảnh JPEG tới chat_id."""
        if not self.session:
            return False
        url = f"{self.api_url}/sendPhoto"
        data = {
            "chat_id": chat_id,
            "caption": caption,
            "parse_mode": "HTML",
        }
        files = {
            "photo": ("yolo_snapshot.jpg", io.BytesIO(photo_bytes), "image/jpeg"),
        }
        try:
            res = self.session.post(url, data=data, files=files, timeout=15.0)
            if res.status_code != 200:
                logger.error(f"Lỗi sendPhoto (HTTP {res.status_code}): {res.text}")
                return False
            return True
        except Exception as e:
            logger.error(f"Ngoại lệ khi sendPhoto tới {chat_id}: {e}")
            return False

    def handle_command(self, chat_id: Any, command_text: str) -> None:
        """Xử lý và phản hồi từng lệnh từ người dùng."""
        cmd = command_text.strip().split()[0].lower() if command_text.strip() else ""

        if not self.is_chat_allowed(chat_id):
            logger.warning(f"Từ chối quyền truy cập cho chat_id={chat_id} với lệnh: {cmd}")
            self.send_message(
                chat_id,
                f"⛔ <b>Bạn không có quyền truy cập bot này.</b>\n"
                f"Chat ID của bạn: <code>{chat_id}</code>\n"
                f"Vui lòng thêm Chat ID này vào biến <code>TELEGRAM_ALLOWED_CHAT_IDS</code> trong file <code>.env</code>.",
            )
            return

        if cmd in ("/start", "/help"):
            self._handle_start(chat_id)
        elif cmd == "/status":
            self._handle_status(chat_id)
        elif cmd == "/photo":
            self._handle_photo(chat_id)
        elif cmd == "/flash":
            self._handle_flash(chat_id)
        else:
            self.send_message(
                chat_id,
                "❓ <b>Lệnh không hợp lệ.</b>\n"
                "Gõ <b>/start</b> để xem danh sách các lệnh được hỗ trợ.",
            )

    def _handle_start(self, chat_id: Any) -> None:
        """Gửi tin nhắn chào mừng và hướng dẫn các lệnh."""
        msg = (
            "🤖 <b>ESP32-CAM Face Mask Detection Bot</b>\n"
            "<i>Hệ thống giám sát khẩu trang chạy mô hình YOLO11n trên GPU RTX 5060</i>\n\n"
            "<b>Các lệnh khả dụng:</b>\n"
            "• <b>/status</b> — Xem số lượng người đeo khẩu trang và FPS thời gian thực\n"
            "• <b>/photo</b>  — Nhận ảnh chụp mới nhất kèm bounding box và nhãn YOLO\n"
            "• <b>/flash</b>  — Bật / Tắt đèn Flash trên ESP32-CAM\n"
            "• <b>/start</b>  — Xem lại hướng dẫn sử dụng này\n\n"
            f"<i>Cổng camera: {getattr(self.worker, 'ip', 'Chưa thiết lập')} | Ngưỡng tươi mới: {self.stale_threshold}s</i>"
        )
        self.send_message(chat_id, msg)

    def _handle_status(self, chat_id: Any) -> None:
        """Thống kê số lượng khẩu trang từ kết quả YOLO mới nhất."""
        if not self.worker:
            self.send_message(chat_id, "⚠️ Pipeline worker chưa được khởi tạo trên server.")
            return

        is_fresh, status, _, reason = self.worker.get_yolo_snapshot(self.stale_threshold)

        if not is_fresh:
            last_ts = status.get("last_capture_time", 0.0)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)) if last_ts > 0 else "Chưa từng nhận"
            msg = (
                "⚠️ <b>Không thể cung cấp dữ liệu thời gian thực!</b>\n\n"
                f"• <b>Trạng thái camera:</b> <code>{status.get('camera_status', 'unknown').upper()}</code>\n"
                f"• <b>Chi tiết:</b> {reason}\n"
                f"• <b>Lần cập nhật cuối:</b> {time_str}\n\n"
                "<i>Hệ thống từ chối dùng kết quả cũ để đảm bảo độ chính xác giám sát.</i>"
            )
            self.send_message(chat_id, msg)
            return

        counts = status.get("counts", {})
        fps = status.get("fps", 0.0)
        camera_ip = status.get("camera_ip", "ESP32-CAM")
        last_ts = status.get("last_capture_time", time.time())
        age = time.time() - last_ts
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)) if last_ts > 0 else "Vừa xong"
        mode_str = str(status.get("inference_mode", "local")).upper()
        if mode_str == "CLOUD":
            c_inf_ms = status.get("cloud_inference_ms", 0.0)
            infer_info = f"AWS EC2 CPU ({c_inf_ms:.1f}ms) [CLOUD]"
        else:
            infer_info = f"NVIDIA RTX 5060 (device={status.get('device', '0')}) [LOCAL]"

        msg = (
            "📊 <b>Thống Kê Nhận Diện Khẩu Trang (YOLO11n):</b>\n\n"
            f"✅ <b>Đeo khẩu trang (mask):</b> {counts.get('mask', 0)}\n"
            f"❌ <b>Không khẩu trang (no_mask):</b> {counts.get('no_mask', 0)}\n"
            f"❓ <b>Chưa rõ (unknown):</b> {counts.get('unknown', 0)}\n"
            f"👥 <b>Tổng số khuôn mặt:</b> {counts.get('total', 0)}\n"
            "───────────────────────\n"
            f"⚡ <b>Tốc độ pipeline:</b> {fps:.1f} FPS\n"
            f"📷 <b>Nguồn camera:</b> <code>{camera_ip}</code>\n"
            f"🕒 <b>Thời điểm ghi nhận:</b> {time_str} (cách đây {age:.1f}s)\n"
            f"💻 <b>Inference:</b> {infer_info}"
        )
        self.send_message(chat_id, msg)

    def _handle_photo(self, chat_id: Any) -> None:
        """Gửi ảnh mới nhất đã được vẽ bounding box và nhãn YOLO."""
        if not self.worker:
            self.send_message(chat_id, "⚠️ Pipeline worker chưa được khởi tạo trên server.")
            return

        is_fresh, status, jpeg_bytes, reason = self.worker.get_yolo_snapshot(self.stale_threshold)

        if not is_fresh or not jpeg_bytes:
            last_ts = status.get("last_capture_time", 0.0)
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_ts)) if last_ts > 0 else "Chưa từng nhận"
            msg = (
                "⚠️ <b>Không thể chụp ảnh thời gian thực!</b>\n\n"
                f"• <b>Nguyên nhân:</b> {reason}\n"
                f"• <b>Trạng thái camera:</b> <code>{status.get('camera_status', 'unknown').upper()}</code>\n"
                f"• <b>Khung hình cuối lúc:</b> {time_str}\n\n"
                "<i>Hệ thống từ chối gửi ảnh cũ để tránh sai lệch thông tin giám sát.</i>"
            )
            self.send_message(chat_id, msg)
            return

        counts = status.get("counts", {})
        fps = status.get("fps", 0.0)
        last_ts = status.get("last_capture_time", time.time())
        time_str = time.strftime("%H:%M:%S", time.localtime(last_ts))

        caption = (
            f"📷 <b>ESP32-CAM YOLO11n Frame</b> ({time_str})\n"
            f"Mask: {counts.get('mask', 0)} | No Mask: {counts.get('no_mask', 0)} | Unknown: {counts.get('unknown', 0)}\n"
            f"FPS: {fps:.1f} • GPU RTX 5060"
        )
        self.send_photo(chat_id, jpeg_bytes, caption)

    def _handle_flash(self, chat_id: Any) -> None:
        """Gọi endpoint /flash trên ESP32-CAM để bật/tắt đèn chiếu sáng."""
        if not self.worker:
            self.send_message(chat_id, "⚠️ Pipeline worker chưa được khởi tạo trên server.")
            return

        success, result_msg = self.worker.toggle_flash()
        if success:
            msg = f"💡 <b>Đèn Flash ESP32-CAM:</b>\n{result_msg}"
        else:
            msg = f"⚠️ <b>Lỗi thao tác đèn Flash:</b>\n{result_msg}"
        self.send_message(chat_id, msg)

    def _poll_updates(self) -> None:
        """Vòng lặp Long Polling nhận tin nhắn từ Telegram Bot API."""
        logger.info(f"Telegram Bot Poller đã bắt đầu. Token: {self.token[:8]}***")
        self.session = requests.Session()

        consecutive_errors = 0

        while self.running:
            url = f"{self.api_url}/getUpdates"
            params = {
                "offset": self.last_update_id + 1,
                "timeout": 10,
                "allowed_updates": ["message"],
            }
            try:
                res = self.session.get(url, params=params, timeout=15.0)

                if res.status_code == 409:
                    logger.error(
                        "[TELEGRAM 409 CONFLICT] Xung đột Telegram Polling! "
                        "Có một tiến trình khác (rất có thể là firmware Arduino ESP32-CAM hoặc một script khác) "
                        "đang cùng gọi getUpdates bằng Bot Token này. "
                        "Giải pháp: Tắt polling trong firmware Arduino hoặc tạo Bot Token mới qua @BotFather cho server Python."
                    )
                    time.sleep(5.0)
                    continue

                if res.status_code != 200:
                    logger.warning(f"getUpdates trả về HTTP {res.status_code}: {res.text}")
                    time.sleep(2.0)
                    continue

                data = res.json()
                if not data.get("ok"):
                    logger.warning(f"getUpdates phản hồi không thành công: {data}")
                    time.sleep(2.0)
                    continue

                updates = data.get("result", [])
                consecutive_errors = 0

                for update in updates:
                    update_id = update.get("update_id", 0)
                    if update_id > self.last_update_id:
                        self.last_update_id = update_id

                    msg = update.get("message")
                    if not msg:
                        continue

                    chat = msg.get("chat", {})
                    chat_id = chat.get("id")
                    text = msg.get("text", "")

                    if chat_id is not None and text:
                        logger.info(f"Nhận lệnh từ chat_id={chat_id}: '{text}'")
                        self.handle_command(chat_id, text)

            except requests.Timeout:
                # Timeout bình thường của long-polling, tiếp tục vòng lặp
                continue
            except (requests.RequestException, Exception) as e:
                if not self.running:
                    break
                consecutive_errors += 1
                sleep_time = min(10.0, 1.0 * consecutive_errors)
                logger.warning(f"Lỗi polling Telegram ({e}). Thử lại sau {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        if self.session:
            self.session.close()
        logger.info("Telegram Bot Poller đã kết thúc.")

    def start(self) -> None:
        """Khởi động luồng polling nền."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._poll_updates, daemon=True, name="TelegramPollerThread")
        self.thread.start()

    def stop(self) -> None:
        """Dừng luồng polling nền an toàn."""
        if not self.running:
            return
        self.running = False
        if self.session:
            try:
                self.session.close()
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        logger.info("Telegram Bot Service đã dừng an toàn.")

