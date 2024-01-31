import torch
import numpy as np
import gluonnlp as nlp

from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

from BERTData import BERTDataset, BERTClassifier

## parameter 설정
max_len = 64  # max seqence length
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

def predict(predict_sentence):
    device = torch.device("cuda")

    learning_rate = 5e-5

    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)

    data = [predict_sentence, 0]
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_loader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_loader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("부정적")
            else:
                test_eval.append("긍정적")

        print(">> 해당 리뷰는 " + test_eval[0] + " 리뷰 입니다.")