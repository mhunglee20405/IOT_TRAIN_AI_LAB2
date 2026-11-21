# -*- coding: utf-8 -*-
"""
tests/test_dashboard_fix.py
Kiểm thử tự động các bản sửa lỗi cho ESP32 Dashboard:
  1. Cấu hình IP mặc định 10.62.83.32 và CLI argument parsing
  2. Trang chính / trả về IP mới
  3. Favicon /favicon.ico trả về 204 No Content
  4. Cú pháp JavaScript hợp lệ (không dính lỗi newline unescaped / AST check qua node --check)
  5. Fallback stream trong JavaScript khi ReadableStream lỗi hoặc không khả dụng
  6. Lazy loading model khi chạy chế độ Cloud
"""

import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".yolo"))

import scripts.esp32_dashboard as dash


class TestDashboardFixes(unittest.TestCase):

    def test_01_default_esp32_ip(self):
        """Kiểm tra DEFAULT_ESP32_IP và tham số CLI --ip mặc định."""
        self.assertEqual(dash.DEFAULT_ESP32_IP, "10.62.83.32")
        sys_argv_bak = sys.argv
        try:
            sys.argv = ["esp32_dashboard.py"]
            parsed = dash.parse_args()
            self.assertEqual(parsed.ip, "10.62.83.32")
        finally:
            sys.argv = sys_argv_bak

    def test_02_javascript_syntax_validity(self):
        """Kiểm tra cú pháp JavaScript trong DASHBOARD_HTML qua node --check."""
        scripts = re.findall(r"<script>(.*?)</script>", dash.DASHBOARD_HTML, re.DOTALL)
        self.assertGreater(len(scripts), 0, "Không tìm thấy thẻ <script> nào trong DASHBOARD_HTML")

        for idx, script_content in enumerate(scripts):
            # Đảm bảo không còn chuỗi `// '\r\n\r\n'` bị phân tách dòng bất hợp lệ
            self.assertNotIn("// '\n\n", script_content)
            self.assertNotIn("// '\r\n\r\n", script_content)

            # Chạy node --check qua file tạm để tương thích tốt nhất trên mọi nền tảng Windows
            tmp_js = ROOT / f"_temp_syntax_test_{idx}.js"
            try:
                tmp_js.write_text(script_content, encoding="utf-8")
                res = subprocess.run(
                    ["node", "--check", str(tmp_js)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    res.returncode,
                    0,
                    f"Script {idx} lỗi cú pháp JS: {res.stderr}",
                )
            finally:
                if tmp_js.exists():
                    tmp_js.unlink()

    def test_03_javascript_stream_fallback_logic(self):
        """Kiểm tra logic fallback sang <img> tag khi ReadableStream gặp sự cố."""
        self.assertIn("fallbackToNativeImg", dash.DASHBOARD_HTML)
        self.assertIn("streamFailCount", dash.DASHBOARD_HTML)
        self.assertIn("fallbackToNativeImg()", dash.DASHBOARD_HTML)
        self.assertIn("mjpegStream", dash.DASHBOARD_HTML)

    def test_04_index_page_and_favicon(self):
        """Kiểm tra hàm xử lý index_page và endpoint /favicon.ico."""
        # 1. Test index_page
        res_index = dash.index_page()
        self.assertEqual(res_index.status_code, 200)
        body_text = res_index.body.decode("utf-8")
        self.assertIn("10.62.83.32", body_text)
        self.assertIn('<span id="cameraIpText">10.62.83.32</span>', body_text)

        # 2. Test favicon endpoint (đồng bộ, trả về 204)
        res_favicon = dash.favicon()
        self.assertEqual(res_favicon.status_code, 204)

    def test_05_lazy_loading_cloud_mode(self):
        """Kiểm tra worker ở chế độ cloud không nạp YOLO model local khi khởi động."""
        worker = dash.ESP32PipelineWorker(
            ip="10.62.83.32",
            model_path=dash.DEFAULT_MODEL_PATH,
            inference_mode="cloud",
        )
        self.assertEqual(worker.inference_mode, "cloud")
        self.assertIsNone(worker.model, "Worker khởi tạo ở chế độ cloud không được nạp local model trước")

        # Kiểm tra switch sang cloud giữ nguyên trạng thái
        ok, msg = worker.set_inference_mode("cloud")
        self.assertTrue(ok)
        self.assertIsNone(worker.model)


if __name__ == "__main__":
    unittest.main()
