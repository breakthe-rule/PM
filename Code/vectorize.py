import numpy as np
from collections import Counter

def vectorize(char_indices,divisor,divisor2,next_chars_t,next_chars,chars,sentences,maxlen,target_chars,target_char_indices,
              sentences_t=[],sentences_t2=[],sentences_t3=[],sentences_t4=[]):
  if len(sentences_t)>0: num_features = len(chars)+5
  else: num_features = len(chars)+1
  X = np.zeros((len(sentences), maxlen, num_features), dtype=np.float32)
  y_a = np.zeros((len(sentences), len(target_chars)), dtype=np.float32)
  y_t = np.zeros((len(sentences)), dtype=np.float32)
  for i, sentence in enumerate(sentences):
      leftpad = maxlen-len(sentence)
      next_t = next_chars_t[i]
      if len(sentences_t)>0:
        sentence_t = sentences_t[i]
        sentence_t2 = sentences_t2[i]
        sentence_t3 = sentences_t3[i]
        sentence_t4 = sentences_t4[i]
      for t, char in enumerate(sentence):
          multiset_abstraction = Counter(sentence[:t+1])
          for c in chars:
              if c==char: #this will encode present events to the right places
                  X[i, t+leftpad, char_indices[c]] = 1
          X[i, t+leftpad, len(chars)] = t+1
          if len(sentences_t)>0:
            X[i, t+leftpad, len(chars)+1] = sentence_t[t]/divisor
            X[i, t+leftpad, len(chars)+2] = sentence_t2[t]/divisor2
            X[i, t+leftpad, len(chars)+3] = sentence_t3[t]/86400
            X[i, t+leftpad, len(chars)+4] = sentence_t4[t]/7
      for c in target_chars:
          if c==next_chars[i]:
              y_a[i, target_char_indices[c]] = 1
      y_t[i] = next_t/divisor
  return X,y_a,y_t