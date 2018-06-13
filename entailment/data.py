import led_parser
import os
import numpy as np

def read_data(fname):
  """
  Reads the data files.
  Args:
    fname (str): the abs or relative path
  """
  with open(fname, 'r') as f:
    data = f.read()
  data = data.split('\n')
  new_data = []
  for d in data[:-1]:
    a, b, e, _, _, _ = tuple(d.split(','))
    new_data.append([a, b, int(e)])
  return np.array(new_data)

def batch_data(data, batch_size):
  n = len(data)
  np.random.shuffle(data)
  data = data.T # transpose the data
  for i in range(n//batch_size-1):
    A = data[0][i*batch_size:(i+1)*batch_size]
    B = data[1][i*batch_size:(i+1)*batch_size]
    E = data[2][i*batch_size:(i+1)*batch_size]
    yield list(A), list(B), list(E.astype(np.float32))

def fetch_data(batch_size):
    # fetch a generator
    fname = '../logical_entailment_dataset/data/train.txt'
    return batch_data(read_data(fname), batch_size)

def fetch_test_sets(path, batch_size):
    fnames = [f for f in os.listdir(path)
              if f.startswith('test')]
    for fname in fnames:
        yield fname, batch_data(read_data(os.path.join(path, fname)), batch_size)

class Parser():
  def __init__(self, language):
    self.language = language
    self.parser = led_parser.Parser(language)
    self.vocabulary = {op: i for i, op in enumerate(language.symbols)}

  def __call__(self, s):
    parse_result = self.parser.parse(s)
    ops = [self.vocabulary[op.decode("utf-8")] for op in parse_result.ops]
    return ops, parse_result.inputs



if __name__ == '__main__':
    gen = fetch_data(50)
    print(next(gen))
