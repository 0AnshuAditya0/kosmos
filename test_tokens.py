from rag.fast_path import _tokens
q = "স্নেয়ার ড্রাম কী?"
print("Query tokens:", [repr(t) for t in _tokens(q)])
passage = "স্নেয়ার ড্রাম: দুটি মাথা এবং নীচের মাথা জুড়ে প্রসারিত একটি"
print("Passage tokens:", [repr(t) for t in _tokens(passage)])