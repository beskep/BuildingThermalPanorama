# 열화상 파노라마를 통한 건물 에너지 검진 결과

## 파노라마

| ![description](image) | ![description](image) |
| --------------------- | --------------------- |
| ![description](image) | ![description](image) |

## 온도 분포

### 온도 보정 인자

| 벽 방사율 | 창문 방사율 |    지점 온도 보정    |
| :-------: | :---------: | :------------------: |
| {e_wall}  | {e_window}  | {delta_temperature}℃ |

### 온도 분포

![description](image)

| 외피부위 |    평균    |  표준편차  |    Q₁     |    중위수     |    Q₃     |
| :------: | :--------: | :--------: | :-------: | :-----------: | :-------: |
|    벽    | {avg_wall} | {std_wall} | {q1_wall} | {median_wall} | {q3_wall} |
|   창문   | {avg_wall} | {std_wall} | {q1_wall} | {median_wall} | {q3_wall} |

## 취약부위 검진 결과

![description](image)
|외부 온도 | 내부 온도 | 벽 취약부위 비율 | 창문 취약부위 비율 |
|:--:|:--:|:--:|:--:|
| {interior_temperature}℃ | {exterior_temperature}℃ | {wall} | {window} |
