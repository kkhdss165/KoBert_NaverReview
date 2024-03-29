# KoBert_NaverReview
- KoBert 활용하여 네이버 리뷰 긍정/부정 판별 모델 학습

### Kobert 설치

- **파이썬 버전 3.7 (Window OS)**
- 깃 불러오기 (SKTBrain/KoBERT)

```bash
git clone 
https://github.com/SKTBrain/KoBERT.git
```

```bash
cd KoBERT
```

- requirements.txt 수정 (의존성 패키지 설치)

```bash
numpy == 1.16.6
boto3 ==1.15.18
gluonnlp == 0.8.0
mxnet == 1.6.0
onnxruntime == 1.8.0
sentencepiece ==0.1.96
torch == 1.10.0
transformers == 4.1.0
```

```bash
pip install -r requirements.txt
```

- Kobert 설치

```bash
pip install .
```

```bash
pip install pandas==0.24.2
```

```bash
pip install scikit-learn==0.20.4
```

```jsx
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://do
wnload.pytorch.org/whl/cu113/torch_stable.html
```