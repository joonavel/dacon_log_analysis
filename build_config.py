with open('C:/Users/joon/dla-flask/dacon_log_analysis/storage/config.yaml', 'w', encoding='utf-8') as f:
    f.write('''
model:
  name: distilbert-base-uncased
comment: null # 기타 추가할 메모
result_dir: /storage # 출력 파일들이 저장될 경로 (*.log 로그 파일과, *.pth 체크포인트가 저장됩니다.)

debug: false
seed: 20210425
ver: 7

train:
  SAM: false
  folds:
    - 1
    - 2
    - 3  # 몇번 째 fold들을 학습할건지. 1~5 사이의 값.
    - 4  # 한 fold마다 GPU에 따라 15~25시간 정도 걸립니다.
    - 5
  checkpoints: # 학습을 checkpoint부터 다시 시작할 때 설정
    - /storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_1.pth
    - /storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_2.pth
    - /storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_3.pth
    - /storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_4.pth
    - /storage/distilbert-base-uncased-focal-AdamW-lr1e-05-ver7-os10_5.pth
  loss:
    name: focal # ce, focal, arcface
    params:
      gamma: 2.0
      s: 45.0
      m: 0.1
      crit: focal

  optimizer:
    name: AdamW # Adam, AdamW

  finetune:
    do: true  # tail부분만 2epochs, body만 2epochs, 전체 8epochs
    step1_epochs: 2
    step2_epochs: 4
  max_epochs: 6

  lr: 0.00001
  scheduler:
    name: ReduceLROnPlateau
    params:
      factor: 0.5
      patience: 3
      verbose: true

dataset:
  dir: data/ver6  # 데이터셋 경로
  batch_size: 32  # Batch size 35에 약 22GB 정도의 GPU 메모리가 필요(finetune-step3 기준)
  num_workers: 8
  oversampling: true  # level 2, 4, 6에 대해서 10배로 데이터를 복제해줍니다.
  oversampling_scale: 10
    ''')

print('Config Created!!')