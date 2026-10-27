# Rock-Paper-Scissors vs Computer (MediaPipe Hand Edition)

Đây là ứng dụng chơi game Oẳn Tù Tì với máy tính, sử dụng Computer Vision để nhận diện cử chỉ bàn tay. 
Dự án được xây dựng với Python, OpenCV và thư viện MediaPipe (sử dụng HandLandmarker Tasks API mới).

## Thuật toán chính
Dự án không sử dụng hoàn toàn module có sẵn của MediaPipe. MediaPipe chỉ cung cấp 21 điểm mốc (landmarks). Các logic cốt lõi do sinh viên tự xây dựng bao gồm:
1. Tính toán vector và tọa độ Y, X của các khớp tay để phân loại hình dạng (Búa, Kéo, Bao).
2. Xây dựng lớp `StabilityTracker` yêu cầu 8 khung hình liên tiếp giống nhau để khử nhiễu.
3. Xây dựng lớp `GameStateMachine` để quản lý luồng thời gian, đếm ngược và kiểm tra thắng thua.

## Cài đặt thư viện cần thiết
Bạn cần cài đặt các thư viện sau thông qua pip:
```bash
pip install opencv-python mediapipe
```
Hướng dẫn sử dụng
Chạy chương trình:

Bash
python rps_mediapipe.py
Lần đầu tiên chạy, chương trình sẽ tự động tải file model hand_landmarker.task từ internet (nếu chưa có).

Phím điều khiển:
Phím SPACE: Bắt đầu một vòng đấu mới. Khi bấm sẽ có 3 giây đếm ngược.
Phím r: Reset (Làm mới) lại điểm số của cả người chơi và máy tính.
Phím q: Thoát ứng dụng.

Video Demo:
https://drive.google.com/file/d/1GMvLw8mrw2gr1JBlLqG1LNHWoe1bL4ux/view?usp=sharing
