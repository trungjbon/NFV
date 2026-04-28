# NFV
## Sử dụng Surrogate-Assisted + NSGA-II + GP để giải bài toán Đặt VNF ​​và Định tuyến lưu lượng trong mạng ảo hóa

- Thực nghiệm trên các cấu trúc liên kết mạng thực tế được sử dụng bao gồm: NSF (14 nút, 21 liên kết), CONUS (75 nút, 99 liên kết), COGENT (104 nút, 116 liên kết)
- Các kết quả thực nghiệm được đánh giá dựa trên 2 tiêu chí reject_rate và cost. Mỗi kết quả được thể hiện theo thứ tự là: +/=/- tương ứng với kết quả GP tốt hơn/bằng/tệ hơn so với các thuật toán heuristic khác.

| Cấu trúc mạng | Greedy (reject_rate +/-/=) | Greedy (cost +/-/=) | First Fit (reject_rate +/-/=) | First Fit (cost +/-/=) | Random (reject_rate +/-/=) | Random (cost +/-/=) |
|--------------|---------------------------|---------------------|-------------------------------|------------------------|-----------------------------|----------------------|
| NSF          | 21 / 3 / 0               | 5 / 3 / 16          | 24 / 0 / 0                    | 9 / 1 / 14             | 24 / 0 / 0                  | 9 / 1 / 14           |
| COGENT       | 23 / 1 / 0               | 0 / 0 / 24          | 24 / 0 / 0                    | 12 / 0 / 12            | 24 / 0 / 0                  | 16 / 0 / 8           |
| CONUS        | 23 / 1 / 0               | 0 / 0 / 24          | 24 / 0 / 0                    | 12 / 0 / 12            | 24 / 0 / 0                  | 13 / 0 / 11          |
| **Tổng**     | **67 / 5 / 0**           | **5 / 3 / 64**      | **72 / 0 / 0**                | **33 / 1 / 38**        | **72 / 0 / 0**              | **38 / 1 / 33**      |
