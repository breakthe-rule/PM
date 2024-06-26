'''
  lines: list of activities in each case
  lines_t: list of list of time difference between consecutive activities in each case
  lines_t2: list of list of time difference between current activity and start of case
  lines_t3: list of list of time difference between current activity and midnight
  lines_t4: list of list of week day (0 monday - 6 sunday) when the activity is performed
  '''

def data(lines,lines_t,lines_t2,lines_t3,lines_t4):
  lines_withend = list(map(lambda x: x+'!',lines))
  
  #List of sub-sequences of events (as strings) for training.
  sentences = []
  #List of the next characters to be predicted after each sub-sequence.
  next_chars = []
  
  #Corresponding values for respective sentences
  sentences_t = []
  sentences_t2 = []
  sentences_t3 = []
  sentences_t4 = []

  # Corresponding values for repective nect_chars
  next_chars_t = []
  next_chars_t2 = []
  next_chars_t3 = []
  next_chars_t4 = []

  for line, line_t, line_t2, line_t3, line_t4 in zip(lines_withend, lines_t, lines_t2, lines_t3, lines_t4):
      for i in range(0, len(line)-1):
          if i==0:
              continue
          #we add iteratively, first symbol of the line, then two first, three...

          sentences.append(line[0: i])
          sentences_t.append(line_t[0:i])
          sentences_t2.append(line_t2[0:i])
          sentences_t3.append(line_t3[0:i])
          sentences_t4.append(line_t4[0:i])

          next_chars.append(line[i])
          next_chars_t.append(line_t[i])
          next_chars_t2.append(line_t2[i])
          next_chars_t3.append(line_t3[i])
          next_chars_t4.append(line_t4[i])

      # Sentence with endline character "!"
      i = len(line)-1
      sentences.append(line[0: i])
      sentences_t.append(line_t[0:i])
      sentences_t2.append(line_t2[0:i])
      sentences_t3.append(line_t3[0:i])
      sentences_t4.append(line_t4[0:i])

      next_chars.append(line[i])
      next_chars_t.append(0)
      next_chars_t2.append(0)
      next_chars_t3.append(0)
      next_chars_t4.append(0)

  '''
  sentences: List of sub-sequences of events (as strings) for training.
  next_chars: Corresponding next charecter to be predicted for each sub-sequence in sentences
  next_chars_t: timestamp of the next event which is to be predected
  '''
  return sentences,sentences_t,sentences_t2,sentences_t3,sentences_t4,next_chars,next_chars_t