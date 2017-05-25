from sklearn.externals import joblib

def save_model(filename, model):
    joblib.dump(model, filename, compress=9)
    
def load_model(filename):
    return joblib.load(filename)

