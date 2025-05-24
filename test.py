import pickle
with open('encodings/encodings.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())  
