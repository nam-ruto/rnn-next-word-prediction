from src.data import prepare_corpus, iter_next_word_batches

text = "I love machine learning.\nI love deep learning."
data_ids, vocab = prepare_corpus(text, min_freq=1, add_bos_eos=True)

for x, y in iter_next_word_batches(data_ids, seq_len=6):
    print(x, "->", y)
    break
