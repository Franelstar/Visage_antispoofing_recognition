import pickle


def predict(filename, features):
    loaded_model = pickle.load(open('models/'+filename, 'rb'))
    return loaded_model.predict(features)